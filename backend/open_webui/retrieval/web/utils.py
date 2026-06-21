import asyncio
import ipaddress
import logging
import socket
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import aiohttp
import aiohttp.resolver
import certifi
import requests
import urllib3.connection
import urllib3.connectionpool
import validators
from requests.adapters import HTTPAdapter
from fastapi.concurrency import run_in_threadpool
from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from open_webui.config import (
    ENABLE_RAG_LOCAL_WEB_FETCH,
    EXTERNAL_WEB_LOADER_API_KEY,
    EXTERNAL_WEB_LOADER_URL,
    FIRECRAWL_API_BASE_URL,
    FIRECRAWL_API_KEY,
    FIRECRAWL_TIMEOUT,
    PLAYWRIGHT_TIMEOUT,
    PLAYWRIGHT_WS_URL,
    TAVILY_API_KEY,
    TAVILY_EXTRACT_DEPTH,
    WEB_FETCH_FILTER_LIST,
    WEB_LOADER_ENGINE,
    WEB_LOADER_TIMEOUT,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import AIOHTTP_CLIENT_ALLOW_REDIRECTS, AIOHTTP_CLIENT_SESSION_SSL, USER_AGENT
from open_webui.retrieval.loaders.external_web import ExternalWebLoader
from open_webui.retrieval.loaders.tavily import TavilyLoader
from open_webui.retrieval.web.firecrawl import scrape_firecrawl_url
from open_webui.retrieval.web import page_cache
from open_webui.utils.misc import is_string_allowed

log = logging.getLogger(__name__)


def resolve_hostname(hostname):
    # Get address information
    addr_info = socket.getaddrinfo(hostname, None)

    # Extract IP addresses from address information
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

    return ipv4_addresses, ipv6_addresses


def validate_url(url: Union[str, Sequence[str]]):
    if isinstance(url, str):
        if isinstance(validators.url(url), validators.ValidationError):
            raise ValueError(ERROR_MESSAGES.INVALID_URL)

        # Reject parser-confusing chars: urlparse and requests/aiohttp split
        # on these differently, e.g. http://127.0.0.1\@1.1.1.1 → urlparse
        # extracts 1.1.1.1 (public, passes filter) while requests connects
        # to 127.0.0.1 (internal). Same shape with tab/CR/LF.
        if any(ch in url for ch in ('\\', '\t', '\n', '\r')):
            log.warning(f'Blocked URL with parser-confusing char: {url!r}')
            raise ValueError(ERROR_MESSAGES.INVALID_URL)

        parsed_url = urllib.parse.urlparse(url)

        # Protocol validation - only allow http/https
        if parsed_url.scheme not in ['http', 'https']:
            log.warning(f'Blocked non-HTTP(S) protocol: {parsed_url.scheme} in URL: {url}')
            raise ValueError(ERROR_MESSAGES.INVALID_URL)

        # Blocklist check using unified filtering logic
        if WEB_FETCH_FILTER_LIST:
            if not is_string_allowed(url, WEB_FETCH_FILTER_LIST):
                log.warning(f'URL blocked by filter list: {url}')
                raise ValueError(ERROR_MESSAGES.INVALID_URL)

        if not ENABLE_RAG_LOCAL_WEB_FETCH:
            # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
            parsed_url = urllib.parse.urlparse(url)
            # Get IPv4 and IPv6 addresses
            ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
            # Check if any of the resolved addresses are private
            # DNS rebinding is mitigated at the connection layer; see _SSRFSafeResolver / _SSRFSafeAdapter
            for ip in ipv4_addresses + ipv6_addresses:
                addr = ipaddress.ip_address(ip)
                if not addr.is_global:
                    raise ValueError(ERROR_MESSAGES.INVALID_URL)
        return True
    elif isinstance(url, Sequence):
        return all(validate_url(u) for u in url)
    else:
        return False


def safe_validate_urls(url: Sequence[str]) -> Sequence[str]:
    valid_urls = []
    for u in url:
        try:
            if validate_url(u):
                valid_urls.append(u)
        except Exception as e:
            log.debug(f'Invalid URL {u}: {str(e)}')
            continue
    return valid_urls


def _ssrf_safe_new_conn(self):
    """Resolve DNS, validate all IPs are global, connect to validated IP.

    Replaces urllib3's _new_conn so the DNS lookup that feeds the actual TCP
    connect is the same one we validate — no second resolution, no rebinding
    window.
    """
    host = getattr(self, '_dns_host', self.host)
    port = self.port
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    if not infos:
        raise OSError(f'getaddrinfo for {host!r} returned empty list')
    if not ENABLE_RAG_LOCAL_WEB_FETCH:
        for _, _, _, _, sa in infos:
            if not ipaddress.ip_address(sa[0]).is_global:
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
    err = None
    for fam, typ, proto, _, sa in infos:
        sock = None
        try:
            sock = socket.socket(fam, typ, proto)
            if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(self.timeout)
            if getattr(self, 'source_address', None):
                sock.bind(self.source_address)
            for opt in getattr(self, 'socket_options', None) or ():
                sock.setsockopt(*opt)
            sock.connect(sa)
            return sock
        except OSError as exc:
            err = exc
            if sock is not None:
                sock.close()
    raise err or OSError(f'connect to {host!r}:{port} failed')


class _SafeHTTPConn(urllib3.connection.HTTPConnection):
    _new_conn = _ssrf_safe_new_conn


class _SafeHTTPSConn(urllib3.connection.HTTPSConnection):
    _new_conn = _ssrf_safe_new_conn


class _SafeHTTPPool(urllib3.connectionpool.HTTPConnectionPool):
    ConnectionCls = _SafeHTTPConn


class _SafeHTTPSPool(urllib3.connectionpool.HTTPSConnectionPool):
    ConnectionCls = _SafeHTTPSConn


class _SSRFSafeAdapter(HTTPAdapter):
    """requests transport adapter that validates resolved IPs at connect time."""

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            'http': _SafeHTTPPool,
            'https': _SafeHTTPSPool,
        }


class _SSRFSafeResolver(aiohttp.resolver.DefaultResolver):
    """aiohttp resolver that rejects non-global IPs unless local fetch is on."""

    async def resolve(self, host, port=0, family=socket.AF_INET):
        results = await super().resolve(host, port, family)
        if not ENABLE_RAG_LOCAL_WEB_FETCH:
            for entry in results:
                if not ipaddress.ip_address(entry['host']).is_global:
                    raise ValueError(ERROR_MESSAGES.INVALID_URL)
        return results


def extract_metadata(soup, url):
    metadata = {'source': url}
    if title := soup.find('title'):
        metadata['title'] = title.get_text()
    if description := soup.find('meta', attrs={'name': 'description'}):
        metadata['description'] = description.get('content', 'No description found.')
    if html := soup.find('html'):
        metadata['language'] = html.get('lang', 'No language found.')
    return metadata


# Bound the pre-flight TLS check. Without a timeout, a single slow / unreachable
# host (e.g. AAAA record with no IPv6 route, ICMP-blackholed firewall) blocks the
# threadpool task indefinitely, which in turn hangs SafePlaywrightURLLoader before
# it ever connects to the browser.
SSL_VERIFY_TIMEOUT = 5.0


def verify_ssl_cert(url: str) -> bool:
    """Verify SSL certificate for the given URL."""
    if not url.startswith('https://'):
        return True

    try:
        hostname = url.split('://')[-1].split('/')[0]
        context = ssl.create_default_context(cafile=certifi.where())
        with socket.create_connection((hostname, 443), timeout=SSL_VERIFY_TIMEOUT) as raw_sock:
            raw_sock.settimeout(SSL_VERIFY_TIMEOUT)
            with context.wrap_socket(raw_sock, server_hostname=hostname):
                return True
    except (socket.timeout, TimeoutError) as e:
        log.warning(
            f'SSL verification timed out for {url} after {SSL_VERIFY_TIMEOUT}s: {e}'
        )
        return False
    except ssl.SSLError as e:
        log.warning(f'SSL verification failed (handshake) for {url}: {e}')
        return False
    except Exception as e:
        log.warning(f'SSL verification failed for {url}: {e}')
        return False


class RateLimitMixin:
    async def _wait_for_rate_limit(self):
        """Wait to respect the rate limit if specified."""
        if self.requests_per_second and self.last_request_time:
            min_interval = timedelta(seconds=1.0 / self.requests_per_second)
            time_since_last = datetime.now() - self.last_request_time
            if time_since_last < min_interval:
                await asyncio.sleep((min_interval - time_since_last).total_seconds())
        self.last_request_time = datetime.now()

    def _sync_wait_for_rate_limit(self):
        """Synchronous version of rate limit wait."""
        if self.requests_per_second and self.last_request_time:
            min_interval = timedelta(seconds=1.0 / self.requests_per_second)
            time_since_last = datetime.now() - self.last_request_time
            if time_since_last < min_interval:
                time.sleep((min_interval - time_since_last).total_seconds())
        self.last_request_time = datetime.now()


class URLProcessingMixin:
    async def _verify_ssl_cert(self, url: str) -> bool:
        """Verify SSL certificate for a URL."""
        return await run_in_threadpool(verify_ssl_cert, url)

    async def _safe_process_url(self, url: str) -> bool:
        """Perform safety checks before processing a URL."""
        if self.verify_ssl and not await self._verify_ssl_cert(url):
            raise ValueError(f'SSL certificate verification failed for {url}')
        await self._wait_for_rate_limit()
        return True

    def _safe_process_url_sync(self, url: str) -> bool:
        """Synchronous version of safety checks."""
        if self.verify_ssl and not verify_ssl_cert(url):
            raise ValueError(f'SSL certificate verification failed for {url}')
        self._sync_wait_for_rate_limit()
        return True


class SafeFireCrawlLoader(BaseLoader, RateLimitMixin, URLProcessingMixin):
    def __init__(
        self,
        web_paths,
        verify_ssl: bool = True,
        trust_env: bool = False,
        requests_per_second: Optional[float] = None,
        continue_on_failure: bool = True,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: Optional[int] = None,
        mode: Literal['crawl', 'scrape', 'map'] = 'scrape',
        proxy: Optional[Dict[str, str]] = None,
        params: Optional[Dict] = None,
    ):
        proxy_server = proxy.get('server') if proxy else None
        if trust_env and not proxy_server:
            env_proxies = urllib.request.getproxies()
            env_proxy_server = env_proxies.get('https') or env_proxies.get('http')
            if env_proxy_server:
                if proxy:
                    proxy['server'] = env_proxy_server
                else:
                    proxy = {'server': env_proxy_server}
        self.web_paths = web_paths
        self.verify_ssl = verify_ssl
        self.requests_per_second = requests_per_second
        self.last_request_time = None
        self.trust_env = trust_env
        self.continue_on_failure = continue_on_failure
        self.api_key = api_key
        self.api_url = (api_url or 'https://api.firecrawl.dev').rstrip('/')
        self.timeout = timeout
        self.mode = mode
        self.params = params or {}

    def lazy_load(self) -> Iterator[Document]:
        try:
            for url in self.web_paths:
                doc = scrape_firecrawl_url(
                    self.api_url,
                    self.api_key,
                    url,
                    verify_ssl=self.verify_ssl,
                    timeout=self.timeout,
                    params=self.params,
                )
                if doc is not None:
                    yield doc
        except Exception as e:
            if self.continue_on_failure:
                log.warning(f'Error extracting content from URLs with Firecrawl: {e}')
            else:
                raise e

    async def alazy_load(self):
        try:
            docs = await run_in_threadpool(lambda: list(self.lazy_load()))
            for doc in docs:
                yield doc
        except Exception as e:
            if self.continue_on_failure:
                log.warning(f'Error extracting content from URLs with Firecrawl: {e}')
            else:
                raise e


class SafeTavilyLoader(BaseLoader, RateLimitMixin, URLProcessingMixin):
    def __init__(
        self,
        web_paths: Union[str, List[str]],
        api_key: str,
        extract_depth: Literal['basic', 'advanced'] = 'basic',
        continue_on_failure: bool = True,
        requests_per_second: Optional[float] = None,
        verify_ssl: bool = True,
        trust_env: bool = False,
        proxy: Optional[Dict[str, str]] = None,
    ):
        """Initialize SafeTavilyLoader with rate limiting and SSL verification support.

        Args:
            web_paths: List of URLs/paths to process.
            api_key: The Tavily API key.
            extract_depth: Depth of extraction ("basic" or "advanced").
            continue_on_failure: Whether to continue if extraction of a URL fails.
            requests_per_second: Number of requests per second to limit to.
            verify_ssl: If True, verify SSL certificates.
            trust_env: If True, use proxy settings from environment variables.
            proxy: Optional proxy configuration.
        """
        # Initialize proxy configuration if using environment variables
        proxy_server = proxy.get('server') if proxy else None
        if trust_env and not proxy_server:
            env_proxies = urllib.request.getproxies()
            env_proxy_server = env_proxies.get('https') or env_proxies.get('http')
            if env_proxy_server:
                if proxy:
                    proxy['server'] = env_proxy_server
                else:
                    proxy = {'server': env_proxy_server}

        # Store parameters for creating TavilyLoader instances
        self.web_paths = web_paths if isinstance(web_paths, list) else [web_paths]
        self.api_key = api_key
        self.extract_depth = extract_depth
        self.continue_on_failure = continue_on_failure
        self.verify_ssl = verify_ssl
        self.trust_env = trust_env
        self.proxy = proxy

        # Add rate limiting
        self.requests_per_second = requests_per_second
        self.last_request_time = None

    def lazy_load(self) -> Iterator[Document]:
        """Load documents with rate limiting support, delegating to TavilyLoader."""
        valid_urls = []
        for url in self.web_paths:
            try:
                self._safe_process_url_sync(url)
                valid_urls.append(url)
            except Exception as e:
                log.warning(f'SSL verification failed for {url}: {str(e)}')
                if not self.continue_on_failure:
                    raise e
        if not valid_urls:
            if self.continue_on_failure:
                log.warning('No valid URLs to process after SSL verification')
                return
            raise ValueError('No valid URLs to process after SSL verification')
        try:
            loader = TavilyLoader(
                urls=valid_urls,
                api_key=self.api_key,
                extract_depth=self.extract_depth,
                continue_on_failure=self.continue_on_failure,
            )
            yield from loader.lazy_load()
        except Exception as e:
            if self.continue_on_failure:
                log.exception(f'Error extracting content from URLs: {e}')
            else:
                raise e

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async version with rate limiting and SSL verification."""
        valid_urls = []
        for url in self.web_paths:
            try:
                await self._safe_process_url(url)
                valid_urls.append(url)
            except Exception as e:
                log.warning(f'SSL verification failed for {url}: {str(e)}')
                if not self.continue_on_failure:
                    raise e

        if not valid_urls:
            if self.continue_on_failure:
                log.warning('No valid URLs to process after SSL verification')
                return
            raise ValueError('No valid URLs to process after SSL verification')

        try:
            loader = TavilyLoader(
                urls=valid_urls,
                api_key=self.api_key,
                extract_depth=self.extract_depth,
                continue_on_failure=self.continue_on_failure,
            )
            async for document in loader.alazy_load():
                yield document
        except Exception as e:
            if self.continue_on_failure:
                log.exception(f'Error loading URLs: {e}')
            else:
                raise e


class SafePlaywrightURLLoader(PlaywrightURLLoader, RateLimitMixin, URLProcessingMixin):
    """Load HTML pages safely with Playwright, supporting SSL verification, rate limiting, and remote browser connection.

    Attributes:
        web_paths (List[str]): List of URLs to load.
        verify_ssl (bool): If True, verify SSL certificates.
        trust_env (bool): If True, use proxy settings from environment variables.
        requests_per_second (Optional[float]): Number of requests per second to limit to.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        headless (bool): If True, the browser will run in headless mode.
        proxy (dict): Proxy override settings for the Playwright session.
        playwright_ws_url (Optional[str]): WebSocket endpoint URI for remote browser connection.
        playwright_timeout (Optional[int]): Maximum operation time in milliseconds.
    """

    def __init__(
        self,
        web_paths: List[str],
        verify_ssl: bool = True,
        trust_env: bool = False,
        requests_per_second: Optional[float] = None,
        continue_on_failure: bool = True,
        headless: bool = True,
        remove_selectors: Optional[List[str]] = None,
        proxy: Optional[Dict[str, str]] = None,
        playwright_ws_url: Optional[str] = None,
        playwright_timeout: Optional[int] = 10000,
        cache_ttl_seconds: Optional[int] = None,
    ):
        """Initialize with additional safety parameters and remote browser support.

        ``cache_ttl_seconds`` overrides the env-configured page-cache TTL for
        this loader instance. ``None`` means "use the cache module default"
        (typically 6h). A non-positive value disables the cache entirely for
        this batch -- useful for callers that want a forced refresh.
        """

        proxy_server = proxy.get('server') if proxy else None
        if trust_env and not proxy_server:
            env_proxies = urllib.request.getproxies()
            env_proxy_server = env_proxies.get('https') or env_proxies.get('http')
            if env_proxy_server:
                if proxy:
                    proxy['server'] = env_proxy_server
                else:
                    proxy = {'server': env_proxy_server}

        # We'll set headless to False if using playwright_ws_url since it's handled by the remote browser
        super().__init__(
            urls=web_paths,
            continue_on_failure=continue_on_failure,
            headless=headless if playwright_ws_url is None else False,
            remove_selectors=remove_selectors,
            proxy=proxy,
        )
        self.verify_ssl = verify_ssl
        self.requests_per_second = requests_per_second
        self.last_request_time = None
        self.playwright_ws_url = playwright_ws_url
        self.trust_env = trust_env
        self.playwright_timeout = playwright_timeout
        self.cache_ttl_seconds = cache_ttl_seconds

    def _intercept_navigation_sync(self, route, request=None):
        req = request or route.request

        if req.resource_type != 'document':
            route.continue_()
            return

        try:
            validate_url(req.url)
        except Exception:
            route.abort()
            return

        if AIOHTTP_CLIENT_ALLOW_REDIRECTS:
            resp = route.fetch()
        else:
            try:
                resp = route.fetch(max_redirects=0)
            except TypeError:
                route.abort()
                return

            if 300 <= resp.status < 400:
                route.abort()
                return

        route.fulfill(response=resp)

    async def _intercept_navigation(self, route, request=None):
        # Late route events for already-closed pages are normal under
        # Playwright: any in-flight subresource (trackers, deferred fetches,
        # late analytics beacons) can fire its route hook AFTER the parent
        # `page.goto()` has already returned and the page has been closed in
        # the alazy_load `finally` block. When that happens, every Playwright
        # call on `route` raises `TargetClosedError: ... Target page, context
        # or browser has been closed`. The right behavior is to drop those
        # late events silently — the request was going to be cancelled with
        # the page anyway. Letting the exception escape causes alazy_load to
        # tag the entire batch as fatally dead and abort all remaining URLs.
        async def _safe_route_call(coro_factory):
            try:
                return await coro_factory()
            except Exception:
                return None

        req = request or route.request

        if req.resource_type != 'document':
            await _safe_route_call(route.continue_)
            return

        try:
            await run_in_threadpool(validate_url, req.url)
        except Exception:
            await _safe_route_call(route.abort)
            return

        try:
            if AIOHTTP_CLIENT_ALLOW_REDIRECTS:
                resp = await route.fetch()
            else:
                try:
                    resp = await route.fetch(max_redirects=0)
                except TypeError:
                    await _safe_route_call(route.abort)
                    return

                if 300 <= resp.status < 400:
                    await _safe_route_call(route.abort)
                    return

            await route.fulfill(response=resp)
        except Exception:
            # Page closed mid-fetch / mid-fulfill. The browser is still
            # alive; the rest of alazy_load can keep going with the next
            # URL. Swallowing here is what prevents one slow tracker from
            # poisoning the whole batch.
            return

    def lazy_load(self) -> Iterator[Document]:
        """Safely load URLs synchronously with support for remote browser."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Use remote browser if ws_endpoint is provided, otherwise use local browser
            if self.playwright_ws_url:
                browser = p.chromium.connect(self.playwright_ws_url)
            else:
                browser = p.chromium.launch(headless=self.headless, proxy=self.proxy)

            for url in self.urls:
                try:
                    self._safe_process_url_sync(url)
                    page = browser.new_page()
                    page.route('**/*', self._intercept_navigation_sync)
                    response = page.goto(
                        url,
                        timeout=self.playwright_timeout,
                        wait_until='domcontentloaded',
                    )
                    if response is None:
                        raise ValueError(f'page.goto() returned None for url {url}')

                    text = self.evaluator.evaluate(page, browser, response)
                    metadata = {'source': url}
                    yield Document(page_content=text, metadata=metadata)
                except Exception as e:
                    if self.continue_on_failure:
                        log.exception(f'Error loading {url}: {e}')
                        continue
                    raise e
            browser.close()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Safely load URLs asynchronously with support for remote browser.

        Loads URLs concurrently inside a single browser connection. Each URL
        runs in its own ``BrowserContext`` so that misbehaving pages cannot
        leak route handlers, cookies, or storage onto siblings -- a behavior
        we directly observed previously where one slow tracker request from
        ``vimm.net`` raised ``TargetClosedError`` and poisoned the entire
        batch under the old sequential ``new_page()``-per-URL implementation.

        Concurrency is bounded by ``self.requests_per_second`` (sourced from
        ``WEB_LOADER_CONCURRENT_REQUESTS``; the upstream env var is poorly
        named but its INTENT was always "max parallel URL fetches" -- we
        finally honor that here). Documents are collected via
        ``asyncio.gather`` and then yielded in URL-input order so downstream
        consumers (citations UI, etc.) see deterministic ordering.
        """
        from playwright.async_api import async_playwright

        # Phase 1 page cache: short-circuit URLs we've already fetched recently.
        # Cached docs are held aside and yielded at the end in original URL
        # order alongside freshly-loaded ones, so downstream consumers (the
        # citations UI) see the same deterministic ordering as before.
        cache_ttl = self.cache_ttl_seconds
        cached_docs: Dict[str, Document] = {}
        if page_cache.is_enabled() and (cache_ttl is None or cache_ttl > 0):
            for url in self.urls:
                content = page_cache.get(url, cache_ttl)
                if content is not None:
                    cached_docs[url] = Document(
                        page_content=content,
                        metadata={'source': url, 'cache_hit': True},
                    )
            if cached_docs:
                log.info(
                    'page_cache: serving %d/%d URLs from cache (ttl=%ss)',
                    len(cached_docs),
                    len(self.urls),
                    cache_ttl if cache_ttl is not None else page_cache.default_ttl_seconds(),
                )

        uncached_urls = [u for u in self.urls if u not in cached_docs]

        # All hits -- skip launching Playwright entirely.
        if not uncached_urls:
            for url in self.urls:
                doc = cached_docs.get(url)
                if doc is not None:
                    yield doc
            return

        # Connection-level failures from the remote browser ("socket hang up",
        # closed/disconnected target, etc.) leave the `browser` handle pointing
        # at a corpse — every subsequent operation queues RPCs against a session
        # that no longer exists, and the final `browser.close()` /
        # async_playwright `__aexit__` block forever waiting for them. Detect
        # those errors and skip remaining tasks early.
        #
        # Deliberately NOT in this list: "target page, context or browser has
        # been closed" — that string is what Playwright raises from inside our
        # `_intercept_navigation` route handler when a late subresource event
        # fires after a `page.goto()` has already finished and the page has
        # been closed (which is normal under Playwright). Treating it as a
        # browser-level fatal was poisoning whole batches over a single late
        # tracker request. "target closed" alone (without "page, context or
        # browser ... closed") is genuinely browser death and stays in.
        FATAL_BROWSER_ERROR_MARKERS = (
            'socket hang up',
            'browser has disconnected',
            'browser closed',
            'connection closed',
            'target closed',
            'session with given id not found',
        )

        # Max URLs in flight at once. ``requests_per_second`` is the only
        # tunable already wired to this class; we reuse it as the concurrency
        # cap. Default 5 keeps memory bounded on the remote chromium even if
        # a knowledge-base ingestion ever hands us hundreds of URLs at once.
        max_concurrent = max(1, int(self.requests_per_second or 5))
        semaphore = asyncio.Semaphore(max_concurrent)

        async with async_playwright() as p:
            # Use remote browser if ws_endpoint is provided, otherwise local.
            if self.playwright_ws_url:
                browser = await p.chromium.connect(self.playwright_ws_url)
            else:
                browser = await p.chromium.launch(headless=self.headless, proxy=self.proxy)

            # ``asyncio.Event`` is set by the first task to observe a fatal
            # browser-level error. Tasks still pending will short-circuit
            # without acquiring the semaphore.
            browser_dead = asyncio.Event()

            async def _load_one(url: str) -> Optional[Document]:
                if browser_dead.is_set():
                    log.warning(
                        'Skipping %s: remote browser session is dead, no point '
                        'trying additional URLs in this batch',
                        url,
                    )
                    return None

                async with semaphore:
                    if browser_dead.is_set():
                        return None

                    context = None
                    page = None
                    try:
                        await self._safe_process_url(url)
                        # Context-per-URL is the isolation boundary that fixes
                        # the "one bad tracker kills the batch" failure mode.
                        # Route handlers, cookies, and any IndexedDB / SW
                        # state are scoped to the context, so siblings can't
                        # see each other's tear-down events.
                        context = await browser.new_context()
                        page = await context.new_page()
                        await page.route('**/*', self._intercept_navigation)
                        response = await page.goto(
                            url,
                            timeout=self.playwright_timeout,
                            wait_until='domcontentloaded',
                        )
                        if response is None:
                            raise ValueError(f'page.goto() returned None for url {url}')

                        text = await self.evaluator.evaluate_async(page, browser, response)
                        return Document(page_content=text, metadata={'source': url})
                    except Exception as e:
                        err_msg = str(e).lower()
                        is_fatal = any(m in err_msg for m in FATAL_BROWSER_ERROR_MARKERS)
                        if is_fatal:
                            browser_dead.set()
                            log.error(
                                'Fatal Playwright connection error while loading %s; '
                                'remaining URLs in this batch will be skipped: %s',
                                url,
                                e,
                            )
                        elif self.continue_on_failure:
                            log.exception(f'Error loading {url}: {e}')
                        else:
                            raise
                        return None
                    finally:
                        # Best-effort cleanup. Closing an already-closed
                        # page/context raises; we deliberately ignore those.
                        # Order matters: close the page before its context so
                        # Playwright doesn't surface a redundant page-close
                        # error during context teardown.
                        for closeable in (page, context):
                            if closeable is None:
                                continue
                            try:
                                await closeable.close()
                            except Exception:
                                pass

            results = await asyncio.gather(
                *(_load_one(url) for url in uncached_urls),
                return_exceptions=False,
            )

            loaded: Dict[str, Document] = {}
            for url, doc in zip(uncached_urls, results):
                if doc is None:
                    continue
                loaded[url] = doc
                # Best-effort cache write; never let a cache failure break
                # the search. ``put`` already swallows + DEBUG-logs, but the
                # outer try keeps us defensive against import-time issues.
                try:
                    page_cache.put(url, doc.page_content)
                except Exception as e:
                    log.debug('page_cache: write failed for %s: %s', url, e)

            for url in self.urls:
                doc = cached_docs.get(url) or loaded.get(url)
                if doc is not None:
                    yield doc

            if not browser_dead.is_set():
                try:
                    await browser.close()
                except Exception as e:
                    log.warning(f'Browser close failed: {e}')


class SafeWebBaseLoader(WebBaseLoader):
    """WebBaseLoader with enhanced error handling for URLs."""

    def __init__(self, trust_env: bool = False, *args, **kwargs):
        """Initialize SafeWebBaseLoader
        Args:
            trust_env (bool, optional): set to True if using proxy to make web requests, for example
                using http(s)_proxy environment variables. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.trust_env = trust_env

        # Propagate USER_AGENT env var so that both the sync _scrape() and
        # async _fetch() paths present a real UA instead of python-requests/2.x
        # which gets blocked by Cloudflare, Wikipedia, and similar bot-detection.
        # _fetch() forwards self.session.headers to the aiohttp session, so
        # setting it here covers both code-paths.
        if USER_AGENT:
            self.session.headers['User-Agent'] = USER_AGENT

        # Prevent redirect-based SSRF on the synchronous _scrape() path.
        # validate_url() is called once on the originally-submitted URL, but the
        # parent WebBaseLoader's _scrape() invokes self.session.get(url, **self.requests_kwargs)
        # which by default follows redirects. Without the override below, an attacker
        # can submit a public URL that 302-redirects to an internal address (RFC1918,
        # 127.0.0.1, 169.254.169.254, etc.) and the redirected target is fetched without
        # re-validation. Matches the policy enforced on the async _fetch() path below.
        self.requests_kwargs = {
            **(self.requests_kwargs or {}),
            'allow_redirects': AIOHTTP_CLIENT_ALLOW_REDIRECTS,
        }

        self.session.mount('http://', _SSRFSafeAdapter())
        self.session.mount('https://', _SSRFSafeAdapter())

    async def _fetch(self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5) -> str:
        connector = aiohttp.TCPConnector(resolver=_SSRFSafeResolver())
        async with aiohttp.ClientSession(trust_env=self.trust_env, connector=connector) as session:
            for i in range(retries):
                try:
                    kwargs: Dict = dict(
                        headers=self.session.headers,
                        cookies=self.session.cookies.get_dict(),
                    )
                    if not self.session.verify:
                        kwargs['ssl'] = False
                    else:
                        kwargs['ssl'] = AIOHTTP_CLIENT_SESSION_SSL

                    async with session.get(
                        url,
                        **(self.requests_kwargs | kwargs),
                    ) as response:
                        if self.raise_for_status:
                            response.raise_for_status()
                        return await response.text()
                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1:
                        raise
                    else:
                        log.warning(f'Error fetching {url} with attempt {i + 1}/{retries}: {e}. Retrying...')
                        await asyncio.sleep(cooldown * backoff**i)
        raise ValueError('retry count exceeded')

    def _unpack_fetch_results(self, results: Any, urls: List[str], parser: Union[str, None] = None) -> List[Any]:
        """Unpack fetch results into BeautifulSoup objects."""
        from bs4 import BeautifulSoup

        final_results = []
        for i, result in enumerate(results):
            url = urls[i]
            if parser is None:
                if url.endswith('.xml'):
                    parser = 'xml'
                else:
                    parser = self.default_parser
                self._check_parser(parser)
            final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))
        return final_results

    async def ascrape_all(self, urls: List[str], parser: Union[str, None] = None) -> List[Any]:
        """Async fetch all urls, then return soups for all results."""
        results = await self.fetch_all(urls)
        return self._unpack_fetch_results(results, urls, parser=parser)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path with error handling."""
        for path in self.web_paths:
            try:
                soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
                text = soup.get_text(**self.bs_get_text_kwargs)

                # Build metadata
                metadata = extract_metadata(soup, path)

                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                # Log the error and continue with the next URL
                log.exception(f'Error loading {path}: {e}')

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy load text from the url(s) in web_path."""
        results = await self.ascrape_all(self.web_paths)
        for path, soup in zip(self.web_paths, results):
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = {'source': path}
            if title := soup.find('title'):
                metadata['title'] = title.get_text()
            if description := soup.find('meta', attrs={'name': 'description'}):
                metadata['description'] = description.get('content', 'No description found.')
            if html := soup.find('html'):
                metadata['language'] = html.get('lang', 'No language found.')
            yield Document(page_content=text, metadata=metadata)

    async def aload(self) -> list[Document]:
        """Load data into Document objects."""
        return [document async for document in self.alazy_load()]


class SafeTrafilaturaLoader(BaseLoader, RateLimitMixin, URLProcessingMixin):
    """Pure-Python loader: aiohttp fetch + trafilatura content extraction.

    Drop-in replacement for ``SafePlaywrightURLLoader`` for sites that don't
    require JavaScript rendering. Trafilatura is dramatically faster (no
    Chromium, no WebSocket round-trip, no per-URL browser context), strips
    boilerplate (nav, footer, cookie banners, comment sections) the way
    ``SafeWebBaseLoader``'s raw BS4 ``get_text()`` cannot, and emits clean
    markdown that downstream RAG chunking can split on real semantic
    boundaries.

    Trade-off: trafilatura does not execute JavaScript. Single-page apps
    that render their content client-side will come back with empty or
    placeholder bodies. Most news sites, blogs, docs, GitHub, Wikipedia,
    Stack Overflow, Hacker News, and old.reddit work fine; modern JS-only
    app shells (X, Discord, etc.) do not. Keep ``SafePlaywrightURLLoader``
    around for those.

    Wires into the same infrastructure the rest of this module uses:

    - ``_SSRFSafeResolver`` on the aiohttp connector + ``_SSRFSafeAdapter``
      on the sync requests path so redirect-based SSRF is blocked.
    - ``USER_AGENT`` so Cloudflare / Wikipedia / similar bot-detection
      doesn't drop the request.
    - ``AIOHTTP_CLIENT_ALLOW_REDIRECTS`` for the redirect policy.
    - ``page_cache.get`` / ``page_cache.put`` for the same 6h on-disk
      page cache the Playwright loader uses, so repeat searches don't
      re-fetch.
    - ``requests_per_second`` (sourced from ``WEB_LOADER_CONCURRENT_REQUESTS``)
      as the asyncio semaphore cap, matching ``SafePlaywrightURLLoader``.
    """

    def __init__(
        self,
        web_paths: List[str],
        verify_ssl: bool = True,
        trust_env: bool = False,
        requests_per_second: Optional[float] = None,
        continue_on_failure: bool = True,
        timeout: Optional[float] = 15.0,
        output_format: Literal['markdown', 'txt', 'html'] = 'markdown',
        cache_ttl_seconds: Optional[int] = None,
    ):
        self.web_paths = web_paths
        self.verify_ssl = verify_ssl
        self.trust_env = trust_env
        self.requests_per_second = requests_per_second
        self.last_request_time = None
        self.continue_on_failure = continue_on_failure
        self.timeout = timeout
        self.output_format = output_format
        self.cache_ttl_seconds = cache_ttl_seconds

    @staticmethod
    def _build_metadata(html: str, url: str) -> Dict[str, Any]:
        """Best-effort trafilatura metadata extraction (title/author/date/desc)."""
        metadata: Dict[str, Any] = {'source': url}
        try:
            import trafilatura

            meta = trafilatura.extract_metadata(html)
            if meta is None:
                return metadata
            if getattr(meta, 'title', None):
                metadata['title'] = meta.title
            if getattr(meta, 'author', None):
                metadata['author'] = meta.author
            if getattr(meta, 'date', None):
                metadata['date'] = meta.date
            if getattr(meta, 'description', None):
                metadata['description'] = meta.description
            if getattr(meta, 'sitename', None):
                metadata['sitename'] = meta.sitename
            if getattr(meta, 'language', None):
                metadata['language'] = meta.language
        except Exception as e:
            log.debug(f'trafilatura: metadata extraction failed for {url}: {e}')
        return metadata

    def _extract(self, html: str, url: str) -> Optional[Document]:
        """HTML → trafilatura → Document. Returns ``None`` on empty extraction."""
        import trafilatura

        try:
            text = trafilatura.extract(
                html,
                url=url,
                output_format=self.output_format,
                # Strip noise. Comment sections are rarely worth ingesting and
                # tank chunk quality with low-signal threads; tables are the
                # opposite -- most docs/wiki pages put structured facts there.
                include_comments=False,
                include_tables=True,
                # Bias toward keeping only confidently-main-content blocks.
                # On RAG ingestion we'd rather lose a borderline paragraph
                # than poison a chunk with navigation cruft.
                favor_precision=True,
                with_metadata=False,
            )
        except Exception as e:
            log.warning(f'trafilatura: extract() failed for {url}: {e}')
            return None

        if not text or not text.strip():
            log.debug(f'trafilatura: empty extraction for {url} (likely JS-rendered)')
            return None

        metadata = self._build_metadata(html, url)
        return Document(page_content=text, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        """Synchronous fetch + extract path. Used by BaseLoader.load()."""
        session = requests.Session()
        if USER_AGENT:
            session.headers['User-Agent'] = USER_AGENT
        session.mount('http://', _SSRFSafeAdapter())
        session.mount('https://', _SSRFSafeAdapter())

        for url in self.web_paths:
            try:
                self._safe_process_url_sync(url)
                resp = session.get(
                    url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    allow_redirects=AIOHTTP_CLIENT_ALLOW_REDIRECTS,
                )
                resp.raise_for_status()
                doc = self._extract(resp.text, url)
                if doc is not None:
                    yield doc
                    try:
                        page_cache.put(url, doc.page_content)
                    except Exception as e:
                        log.debug('page_cache: write failed for %s: %s', url, e)
            except Exception as e:
                if self.continue_on_failure:
                    log.exception(f'Error loading {url}: {e}')
                    continue
                raise

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async concurrent fetch + extract. Order-preserving with page_cache."""
        cache_ttl = self.cache_ttl_seconds
        cached_docs: Dict[str, Document] = {}
        if page_cache.is_enabled() and (cache_ttl is None or cache_ttl > 0):
            for url in self.web_paths:
                content = page_cache.get(url, cache_ttl)
                if content is not None:
                    cached_docs[url] = Document(
                        page_content=content,
                        metadata={'source': url, 'cache_hit': True},
                    )
            if cached_docs:
                log.info(
                    'page_cache: serving %d/%d URLs from cache (ttl=%ss)',
                    len(cached_docs),
                    len(self.web_paths),
                    cache_ttl if cache_ttl is not None else page_cache.default_ttl_seconds(),
                )

        uncached_urls = [u for u in self.web_paths if u not in cached_docs]
        if not uncached_urls:
            for url in self.web_paths:
                doc = cached_docs.get(url)
                if doc is not None:
                    yield doc
            return

        max_concurrent = max(1, int(self.requests_per_second or 5))
        semaphore = asyncio.Semaphore(max_concurrent)

        headers: Dict[str, str] = {}
        if USER_AGENT:
            headers['User-Agent'] = USER_AGENT

        ssl_arg: Any = AIOHTTP_CLIENT_SESSION_SSL if self.verify_ssl else False
        connector = aiohttp.TCPConnector(resolver=_SSRFSafeResolver())
        client_timeout = aiohttp.ClientTimeout(total=self.timeout) if self.timeout else None

        async def _fetch_and_extract(
            session: aiohttp.ClientSession, url: str
        ) -> Optional[Document]:
            async with semaphore:
                try:
                    await self._safe_process_url(url)
                    async with session.get(
                        url,
                        ssl=ssl_arg,
                        allow_redirects=AIOHTTP_CLIENT_ALLOW_REDIRECTS,
                    ) as response:
                        response.raise_for_status()
                        html = await response.text()
                except Exception as e:
                    if self.continue_on_failure:
                        log.warning(f'Error fetching {url}: {e}')
                        return None
                    raise

                # Push the (CPU-bound) lxml parse off the event loop.
                try:
                    return await run_in_threadpool(self._extract, html, url)
                except Exception as e:
                    if self.continue_on_failure:
                        log.exception(f'Error extracting {url}: {e}')
                        return None
                    raise

        async with aiohttp.ClientSession(
            trust_env=self.trust_env,
            connector=connector,
            headers=headers,
            timeout=client_timeout,
        ) as session:
            results = await asyncio.gather(
                *(_fetch_and_extract(session, url) for url in uncached_urls),
                return_exceptions=False,
            )

        loaded: Dict[str, Document] = {}
        for url, doc in zip(uncached_urls, results):
            if doc is None:
                continue
            loaded[url] = doc
            try:
                page_cache.put(url, doc.page_content)
            except Exception as e:
                log.debug('page_cache: write failed for %s: %s', url, e)

        for url in self.web_paths:
            doc = cached_docs.get(url) or loaded.get(url)
            if doc is not None:
                yield doc

    async def aload(self) -> list[Document]:
        return [document async for document in self.alazy_load()]


def get_web_loader(
    urls: Union[str, Sequence[str]],
    verify_ssl: bool = True,
    requests_per_second: int = 2,
    trust_env: bool = False,
    cache_ttl_seconds: Optional[int] = None,
):
    # Check if the URLs are valid
    safe_urls = safe_validate_urls([urls] if isinstance(urls, str) else urls)

    if not safe_urls:
        log.warning(f'All provided URLs were blocked or invalid: {urls}')
        raise ValueError(ERROR_MESSAGES.INVALID_URL)

    web_loader_args = {
        'web_paths': safe_urls,
        'verify_ssl': verify_ssl,
        'requests_per_second': requests_per_second,
        'continue_on_failure': True,
        'trust_env': trust_env,
    }

    if WEB_LOADER_ENGINE.value == '' or WEB_LOADER_ENGINE.value == 'safe_web':
        WebLoaderClass = SafeWebBaseLoader

        request_kwargs = {}
        if WEB_LOADER_TIMEOUT.value:
            try:
                timeout_value = float(WEB_LOADER_TIMEOUT.value)
            except ValueError:
                timeout_value = None

            if timeout_value:
                request_kwargs['timeout'] = timeout_value

        if request_kwargs:
            web_loader_args['requests_kwargs'] = request_kwargs

    if WEB_LOADER_ENGINE.value == 'playwright':
        WebLoaderClass = SafePlaywrightURLLoader
        web_loader_args['playwright_timeout'] = PLAYWRIGHT_TIMEOUT.value
        if PLAYWRIGHT_WS_URL.value:
            web_loader_args['playwright_ws_url'] = PLAYWRIGHT_WS_URL.value
        # Page cache is honored by the engines that fetch the raw page
        # bytes themselves (Playwright, trafilatura). Firecrawl / Tavily /
        # external already proxy through their own caches and we don't
        # want to second-guess their request semantics.
        if cache_ttl_seconds is not None:
            web_loader_args['cache_ttl_seconds'] = cache_ttl_seconds

    if WEB_LOADER_ENGINE.value == 'trafilatura':
        WebLoaderClass = SafeTrafilaturaLoader
        if WEB_LOADER_TIMEOUT.value:
            try:
                timeout_value = float(WEB_LOADER_TIMEOUT.value)
            except ValueError:
                timeout_value = None
            if timeout_value:
                web_loader_args['timeout'] = timeout_value
        if cache_ttl_seconds is not None:
            web_loader_args['cache_ttl_seconds'] = cache_ttl_seconds

    if WEB_LOADER_ENGINE.value == 'firecrawl':
        WebLoaderClass = SafeFireCrawlLoader
        web_loader_args['api_key'] = FIRECRAWL_API_KEY.value
        web_loader_args['api_url'] = FIRECRAWL_API_BASE_URL.value
        if FIRECRAWL_TIMEOUT.value:
            try:
                web_loader_args['timeout'] = int(FIRECRAWL_TIMEOUT.value)
            except ValueError:
                pass

    if WEB_LOADER_ENGINE.value == 'tavily':
        WebLoaderClass = SafeTavilyLoader
        web_loader_args['api_key'] = TAVILY_API_KEY.value
        web_loader_args['extract_depth'] = TAVILY_EXTRACT_DEPTH.value

    if WEB_LOADER_ENGINE.value == 'external':
        WebLoaderClass = ExternalWebLoader
        web_loader_args['external_url'] = EXTERNAL_WEB_LOADER_URL.value
        web_loader_args['external_api_key'] = EXTERNAL_WEB_LOADER_API_KEY.value

    if WebLoaderClass:
        web_loader = WebLoaderClass(**web_loader_args)

        log.debug(
            'Using WEB_LOADER_ENGINE %s for %s URLs',
            web_loader.__class__.__name__,
            len(safe_urls),
        )

        return web_loader
    else:
        raise ValueError(
            f'Invalid WEB_LOADER_ENGINE: {WEB_LOADER_ENGINE.value}. '
            "Please set it to 'safe_web', 'playwright', 'trafilatura', "
            "'firecrawl', 'tavily', or 'external'."
        )
