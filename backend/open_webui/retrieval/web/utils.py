import asyncio
import ipaddress
import logging
import re
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
        query: Optional[str] = None,
        heading_trim_enabled: bool = True,
        heading_trim_min_token_len: int = 3,
        heading_trim_keep_intro: bool = True,
        js_fallback_enabled: bool = True,
        js_fallback_min_extract_chars: int = 200,
        playwright_ws_url: Optional[str] = None,
        playwright_timeout: Optional[int] = 15000,
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
        # Heading-trim plumbing. ``query`` is the joined-by-space union of
        # every sub-query in the current batch -- so when the same loader
        # instance services a 3-query fanout, every heading is matched
        # against the union of all relevant terms (a page that only matches
        # one of the three queries still keeps its relevant subtrees).
        self.query = query or ''
        self.heading_trim_enabled = heading_trim_enabled
        self.heading_trim_min_token_len = max(1, heading_trim_min_token_len)
        self.heading_trim_keep_intro = heading_trim_keep_intro
        # JS-shell Playwright fallback. ``playwright_ws_url`` carries the
        # ws:// endpoint of the sidecar (PLAYWRIGHT_WS_URL env) and stays
        # ``None`` in tests / local dev where there's no sidecar reachable;
        # the fallback short-circuits to "trafilatura only" in that case
        # regardless of ``js_fallback_enabled``.
        self.js_fallback_enabled = js_fallback_enabled
        self.js_fallback_min_extract_chars = max(0, js_fallback_min_extract_chars)
        self.playwright_ws_url = playwright_ws_url or None
        self.playwright_timeout = playwright_timeout

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

    # Markdown heading detector. Matches ATX-style ``# Heading`` / ``## H``
    # at column zero only — trafilatura's markdown output never indents
    # headings, and rejecting indented lines avoids matching ``#foo`` in
    # comments or hash-prefixed lines inside fenced code blocks.
    _HEADING_RE = re.compile(r'^(#{1,6})\s+(.*\S)\s*$', re.MULTILINE)

    # SPA-shell fingerprints. These are the bare-mount divs that React /
    # Vue / Next / Nuxt apps render server-side as the only meaningful
    # body content -- everything else gets injected client-side by JS we
    # never executed. Substring match (case-insensitive) keeps this
    # robust against minor markup variations across framework versions.
    _SPA_ROOT_HINTS = (
        '<div id="root">',
        "<div id='root'>",
        '<div id="app">',
        "<div id='app'>",
        '<div id="__next">',
        "<div id='__next'>",
        '<div id="__nuxt">',
        "<div id='__nuxt'>",
    )

    # Minimum visible-text-to-HTML ratio. SPA shells are mostly inline
    # script + framework markup; a ratio under 5% almost always means
    # there's no real text on the server-rendered page.
    _SPA_TEXT_RATIO_FLOOR = 0.05

    @staticmethod
    def _looks_js_shell(html: str, extracted_text: str, min_extract_chars: int) -> bool:
        """Return True iff the HTML looks like an unexecuted SPA shell.

        Triggered when trafilatura came back with empty / very thin
        content AND the HTML carries one of the canonical SPA mount
        markers, OR has a substantial <noscript> block, OR the
        text-to-HTML ratio inside <body> is small enough that no
        meaningful prose could have been server-rendered.
        """
        if not html:
            return False

        # Quick reject: trafilatura already produced enough content,
        # so the page isn't a shell even if it also happens to embed
        # a framework mount div somewhere (e.g. a docs page that ships
        # a React-powered search widget alongside its real prose).
        if extracted_text and len(extracted_text) >= min_extract_chars:
            return False

        html_lower = html.lower()
        if any(hint in html_lower for hint in SafeTrafilaturaLoader._SPA_ROOT_HINTS):
            return True

        # <noscript> with substantial content is a strong human-facing
        # tell ("This site requires JavaScript", "Please enable JS"),
        # so we treat any noscript block > 50 chars as a shell marker.
        ns_match = re.search(
            r'<noscript[^>]*>(.*?)</noscript>', html_lower, re.DOTALL
        )
        if ns_match and len(ns_match.group(1).strip()) > 50:
            return True

        # Text-to-HTML ratio inside <body>. We approximate "text"
        # by stripping all tags from the body slice -- crude but
        # adequate as a tie-breaker; we already require an empty /
        # near-empty trafilatura extract to even get here.
        body_match = re.search(
            r'<body[^>]*>(.*?)</body>', html_lower, re.DOTALL
        )
        if body_match:
            body_html = body_match.group(1)
            stripped = re.sub(r'<[^>]+>', '', body_html)
            if body_html and (len(stripped.strip()) / max(len(body_html), 1)) < SafeTrafilaturaLoader._SPA_TEXT_RATIO_FLOOR:
                return True

        return False

    @staticmethod
    def _tokenize_query(query: str, min_len: int) -> List[str]:
        """Split a query into lowercased alnum tokens, dropping short stopwordy ones.

        ``min_len`` filters out 1-2 char tokens like ``is`` / ``to`` / ``a``
        that match almost every English heading and would reduce the trim
        to a no-op. We keep the order-insensitive ``set`` semantics: any
        single token match keeps a subtree.
        """
        if not query:
            return []
        tokens = [t.lower() for t in re.split(r'[^A-Za-z0-9]+', query) if t]
        return [t for t in tokens if len(t) >= min_len]

    def _heading_trim(self, text: str, query: str) -> str:
        """Keep only heading subtrees whose heading or body matches ``query``.

        Algorithm:
          1. Parse all H1..H6 headings (``re.MULTILINE`` over the markdown).
          2. For each heading, compute the subtree slice (heading line up to
             the next heading whose depth is <= the current heading's
             depth). This is the structural unit we keep or drop together.
          3. Tokenize the query; tokens shorter than the configured floor
             are discarded so common stopwords don't blanket-match.
          4. A subtree is "kept" if ANY query token appears (case-insensitive
             substring) in either the heading text OR the subtree body.
          5. Optionally ALWAYS keep the page intro: everything up to the
             first heading (page title + lede paragraph if there is one),
             plus the first heading's own body when it's the H1, capped at
             ~500 chars so a giant intro doesn't defeat the trim. This is
             the safety net that prevents a doc with zero token matches
             from collapsing to an empty string and disappearing from the
             citation list entirely.

        Worst-case shrinkage: a long Wikipedia or RTD page with 30+
        sections typically drops to 20-50% of its original length (the
        relevant 2-4 sections plus the intro); a short blog post or
        landing page with all content under H1 or no headings at all
        stays roughly unchanged. We deliberately tolerate the no-op
        case rather than aggressively chopping short pages where every
        section might be relevant context.
        """
        if not text or not query:
            return text

        tokens = self._tokenize_query(query, self.heading_trim_min_token_len)
        if not tokens:
            return text

        matches = list(self._HEADING_RE.finditer(text))
        if not matches:
            # No structural headings to anchor against -- nothing to trim.
            return text

        # ``before_first`` is the page intro slice (anything before the
        # first heading line). On most clean docs this is the page title
        # markdown that trafilatura emits before any H1 (or the lede
        # paragraph on news-style pages whose H1 was already consumed as
        # title metadata).
        first_start = matches[0].start()
        before_first = text[:first_start]

        # Pair each heading with its subtree end: the start of the next
        # heading at depth <= current depth. We walk linearly so this is
        # O(N) in heading count.
        subtrees: List[tuple[int, int, int, str, str]] = []
        for i, m in enumerate(matches):
            depth = len(m.group(1))
            heading_text = m.group(2)
            start = m.start()
            end = len(text)
            for j in range(i + 1, len(matches)):
                next_depth = len(matches[j].group(1))
                if next_depth <= depth:
                    end = matches[j].start()
                    break
            subtrees.append((start, end, depth, heading_text, text[start:end]))

        # Match each subtree. Substring match is the cheapest reliable
        # check and matches the user's mental model ("the page mentions
        # the word `gemma3`"). Lowercase once per body, not per token.
        kept_ranges: List[tuple[int, int]] = []
        if self.heading_trim_keep_intro and before_first.strip():
            # Always keep page intro slice. Cap at 500 chars so an
            # introductory wall-of-text on a blog post doesn't dilute the
            # trim's benefit.
            kept_ranges.append((0, min(len(before_first), 500)))

        for start, end, _, heading_text, body in subtrees:
            haystack_heading = heading_text.lower()
            haystack_body = body.lower()
            if any(t in haystack_heading or t in haystack_body for t in tokens):
                kept_ranges.append((start, end))
            elif (
                self.heading_trim_keep_intro
                and start == first_start
                and not kept_ranges
            ):
                # If we somehow ended up with zero kept ranges and no
                # intro slice, take the first heading's subtree (capped
                # at ~500 chars of body) as a last-resort fallback so
                # the doc isn't empty. This branch is rare -- it only
                # triggers when ``before_first`` is whitespace-only and
                # nothing matches.
                kept_ranges.append((start, min(end, start + 500)))

        if not kept_ranges:
            # Pure no-match doc and keep-intro disabled. Return original
            # rather than empty -- chunk D's compress pass is downstream
            # and is better positioned to drop irrelevant content than
            # we are here. ``return text`` is also the safe default if
            # all our heuristics misfire.
            return text

        # Merge overlapping / adjacent ranges so we don't double-emit
        # bytes when a kept subtree slot abuts another kept subtree.
        kept_ranges.sort()
        merged: List[tuple[int, int]] = []
        for s, e in kept_ranges:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        return '\n\n'.join(text[s:e].strip() for s, e in merged if s < e)

    def _extract(self, html: str, url: str) -> Optional[Document]:
        """HTML → trafilatura → Document. Returns ``None`` on empty extraction.

        Post-processing: if ``heading_trim_enabled`` is on and the loader
        was constructed with a ``query``, walks the markdown headings and
        keeps only subtrees that match query tokens (see ``_heading_trim``
        for the algorithm). Trafilatura's ``favor_precision=True`` already
        drops the obvious nav/footer chrome; this is the second-pass cut
        that targets *topically irrelevant* subsections (release notes,
        unrelated docs entries, off-topic stack overflow side threads)
        which survived the first pass because they LOOK like real content.
        """
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
            # When the page looks like an unexecuted SPA shell, return a
            # sentinel Document with ``needs_js=True`` so alazy_load can
            # collect the URL and retry it via Playwright. We deliberately
            # do NOT cache this sentinel in page_cache -- the cache is a
            # real-content store and we don't want a future search to
            # serve a stale "needs JS" placeholder.
            if self.js_fallback_enabled and self._looks_js_shell(
                html, '', self.js_fallback_min_extract_chars
            ):
                metadata = self._build_metadata(html, url)
                metadata['needs_js'] = True
                return Document(page_content='', metadata=metadata)
            return None

        # Trafilatura returned content but it's still tiny -- on SPAs that
        # ship a server-rendered fragment of the eventual page (e.g. a
        # title + a "Loading..." placeholder), the extract is non-empty
        # but useless. Flag those too so the Playwright fallback can try
        # to do better.
        if (
            self.js_fallback_enabled
            and len(text.strip()) < self.js_fallback_min_extract_chars
            and self._looks_js_shell(html, text, self.js_fallback_min_extract_chars)
        ):
            metadata = self._build_metadata(html, url)
            metadata['needs_js'] = True
            # Keep the short body too -- if Playwright fails downstream
            # we'd rather have the stub than nothing.
            return Document(page_content=text, metadata=metadata)

        if self.heading_trim_enabled and self.query and self.output_format == 'markdown':
            try:
                original_len = len(text)
                text = self._heading_trim(text, self.query)
                if original_len and len(text) < original_len:
                    log.debug(
                        'heading_trim: %s shrunk from %d to %d chars (%.0f%%)',
                        url,
                        original_len,
                        len(text),
                        100.0 * len(text) / original_len,
                    )
            except Exception as e:
                # Don't poison a fetched page on a regex/parse edge case.
                log.debug('heading_trim: skipped for %s due to error: %s', url, e)

        if not text or not text.strip():
            # Defensive: if the trim somehow emptied the doc, fall back
            # to None so we don't surface a phantom empty citation.
            log.debug(f'trafilatura: heading_trim left {url} empty; dropping')
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
        """Async concurrent fetch + extract. Order-preserving with page_cache.

        Cache strategy is three-tiered:

        1. Fresh cache hit -- ``get_with_validators`` returns content
           and we short-circuit the HTTP request entirely.
        2. Stale-with-validators -- we have an ``ETag`` /
           ``Last-Modified`` from the previous fetch but the TTL has
           expired. The GET below sends those as ``If-None-Match`` /
           ``If-Modified-Since``; on a ``304 Not Modified`` we resurrect
           the cached body via ``get_force`` + ``touch``, paying a tiny
           round-trip instead of re-downloading.
        3. Cache miss -- fall through to the normal fetch and capture
           the response's ``ETag`` / ``Last-Modified`` into the cache
           on success.
        """
        cache_ttl = self.cache_ttl_seconds
        cached_docs: Dict[str, Document] = {}
        # ``stale_validators`` carries the headers we want to send on
        # the conditional GET for URLs whose body is stale but
        # validatable. Empty for fresh hits and pure cache misses.
        stale_validators: Dict[str, Dict[str, str]] = {}
        if page_cache.is_enabled() and (cache_ttl is None or cache_ttl > 0):
            for url in self.web_paths:
                content, etag, last_modified = page_cache.get_with_validators(
                    url, cache_ttl
                )
                if content is not None:
                    cached_docs[url] = Document(
                        page_content=content,
                        metadata={'source': url, 'cache_hit': True},
                    )
                elif etag or last_modified:
                    headers: Dict[str, str] = {}
                    if etag:
                        headers['If-None-Match'] = etag
                    if last_modified:
                        headers['If-Modified-Since'] = last_modified
                    stale_validators[url] = headers
            if cached_docs:
                log.info(
                    'page_cache: serving %d/%d URLs from cache (ttl=%ss)',
                    len(cached_docs),
                    len(self.web_paths),
                    cache_ttl if cache_ttl is not None else page_cache.default_ttl_seconds(),
                )
            if stale_validators:
                log.debug(
                    'page_cache: %d URL(s) stale-but-validatable; '
                    'attempting conditional GETs',
                    len(stale_validators),
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

        base_headers: Dict[str, str] = {}
        if USER_AGENT:
            base_headers['User-Agent'] = USER_AGENT

        ssl_arg: Any = AIOHTTP_CLIENT_SESSION_SSL if self.verify_ssl else False
        connector = aiohttp.TCPConnector(resolver=_SSRFSafeResolver())
        client_timeout = aiohttp.ClientTimeout(total=self.timeout) if self.timeout else None

        async def _fetch_and_extract(
            session: aiohttp.ClientSession, url: str
        ) -> Optional[Document]:
            async with semaphore:
                # Merge per-request validator headers (If-None-Match /
                # If-Modified-Since) on top of the session-level
                # User-Agent. Build a fresh dict so we never mutate
                # the session headers across concurrent requests.
                req_headers = dict(base_headers)
                req_headers.update(stale_validators.get(url, {}))
                try:
                    await self._safe_process_url(url)
                    async with session.get(
                        url,
                        ssl=ssl_arg,
                        allow_redirects=AIOHTTP_CLIENT_ALLOW_REDIRECTS,
                        headers=req_headers,
                    ) as response:
                        # 304 Not Modified -- the server confirmed our
                        # cached body is still valid. Pull it back via
                        # the TTL-ignoring force-get and touch the
                        # freshness timestamp so the next search inside
                        # the TTL window is a plain cache hit.
                        if response.status == 304:
                            cached = page_cache.get_force(url)
                            if cached is not None:
                                page_cache.touch(url)
                                log.debug(
                                    'page_cache: 304 revalidated %s '
                                    '(saved a full fetch)',
                                    url,
                                )
                                return Document(
                                    page_content=cached,
                                    metadata={
                                        'source': url,
                                        'cache_hit': True,
                                        'revalidated': True,
                                    },
                                )
                            # Unusual: server returned 304 but we no
                            # longer have the body (e.g. cache was
                            # cleared between the validator-read above
                            # and this response). Fall back to a fresh
                            # GET without validators.
                            log.debug(
                                'page_cache: 304 for %s but no cached body; '
                                'doing unconditional re-fetch',
                                url,
                            )
                            async with session.get(
                                url,
                                ssl=ssl_arg,
                                allow_redirects=AIOHTTP_CLIENT_ALLOW_REDIRECTS,
                                headers=base_headers,
                            ) as r2:
                                r2.raise_for_status()
                                html = await r2.text()
                                response_etag = r2.headers.get('ETag')
                                response_last_modified = r2.headers.get('Last-Modified')
                        else:
                            response.raise_for_status()
                            html = await response.text()
                            response_etag = response.headers.get('ETag')
                            response_last_modified = response.headers.get('Last-Modified')
                except Exception as e:
                    if self.continue_on_failure:
                        log.warning(f'Error fetching {url}: {e}')
                        return None
                    raise

                # Push the (CPU-bound) lxml parse off the event loop.
                try:
                    doc = await run_in_threadpool(self._extract, html, url)
                except Exception as e:
                    if self.continue_on_failure:
                        log.exception(f'Error extracting {url}: {e}')
                        return None
                    raise
                # Stash the validators on the doc metadata so the
                # caller's cache-write below can persist them alongside
                # the content. Doing it via metadata keeps
                # _fetch_and_extract's return type unchanged.
                if doc is not None:
                    if response_etag:
                        doc.metadata['_response_etag'] = response_etag
                    if response_last_modified:
                        doc.metadata['_response_last_modified'] = response_last_modified
                return doc

        async with aiohttp.ClientSession(
            trust_env=self.trust_env,
            connector=connector,
            headers=base_headers,
            timeout=client_timeout,
        ) as session:
            results = await asyncio.gather(
                *(_fetch_and_extract(session, url) for url in uncached_urls),
                return_exceptions=False,
            )

        loaded: Dict[str, Document] = {}
        needs_js_urls: List[str] = []
        for url, doc in zip(uncached_urls, results):
            if doc is None:
                continue
            loaded[url] = doc
            # Strip the private response-validator keys out of metadata
            # before they hit downstream consumers (citations UI / vector
            # store). They're only useful to the cache write below.
            response_etag = doc.metadata.pop('_response_etag', None)
            response_last_modified = doc.metadata.pop('_response_last_modified', None)
            # Sentinel "needs JS" docs are kept in ``loaded`` only as a
            # fallback in case Playwright also fails. We skip caching
            # them so a future search re-attempts the full pipeline
            # rather than serving the stub from cache. We also skip
            # docs returned from the 304 path, which already came out
            # of the cache (``revalidated=True``).
            if doc.metadata.get('needs_js'):
                needs_js_urls.append(url)
                continue
            if doc.metadata.get('revalidated'):
                continue
            try:
                page_cache.put(
                    url,
                    doc.page_content,
                    etag=response_etag,
                    last_modified=response_last_modified,
                )
            except Exception as e:
                log.debug('page_cache: write failed for %s: %s', url, e)

        # JS-shell fallback. When trafilatura returned empty / shell-only
        # docs AND a Playwright sidecar is reachable AND the fallback is
        # enabled, re-fetch just those URLs through Playwright so the JS
        # actually runs. The new docs replace the trafilatura sentinels
        # in the final result; if Playwright also fails to extract, the
        # original sentinel doc stays (potentially with an empty body)
        # so the URL still appears in the citation list with at least a
        # title.
        if (
            self.js_fallback_enabled
            and needs_js_urls
            and self.playwright_ws_url
        ):
            log.info(
                'trafilatura_js_fallback: %d/%d URLs retried via playwright',
                len(needs_js_urls),
                len(uncached_urls),
            )
            try:
                pw_loader = SafePlaywrightURLLoader(
                    web_paths=needs_js_urls,
                    verify_ssl=self.verify_ssl,
                    trust_env=self.trust_env,
                    # Carry the trafilatura loader's rate-limit / cache
                    # settings forward so the Playwright pass respects
                    # the same SafePlaywrightURLLoader semantics it would
                    # if called directly via get_web_loader.
                    requests_per_second=self.requests_per_second,
                    continue_on_failure=self.continue_on_failure,
                    playwright_ws_url=self.playwright_ws_url,
                    playwright_timeout=self.playwright_timeout,
                    cache_ttl_seconds=self.cache_ttl_seconds,
                )
                pw_docs: List[Document] = []
                async for pw_doc in pw_loader.alazy_load():
                    pw_docs.append(pw_doc)
                # SafePlaywrightURLLoader's alazy_load handles its own
                # cache writes for the success cases; we only need to
                # merge the resulting docs into our final result map.
                for pw_doc in pw_docs:
                    src = pw_doc.metadata.get('source')
                    if not src:
                        continue
                    # Apply the heading trim to Playwright output too --
                    # most Playwright results are plain text (not
                    # markdown), so the trim degrades to a no-op when no
                    # headings are found, which is exactly the safe
                    # behavior we want.
                    if self.heading_trim_enabled and self.query:
                        try:
                            pw_doc.page_content = self._heading_trim(
                                pw_doc.page_content, self.query
                            )
                        except Exception as e:
                            log.debug(
                                'heading_trim: skipped Playwright fallback for %s: %s',
                                src, e,
                            )
                    loaded[src] = pw_doc
            except Exception as e:
                # A broken Playwright sidecar should not poison the
                # batch -- we already have trafilatura sentinels in
                # ``loaded`` for these URLs, so worst case is they
                # surface with empty bodies (URL + title only).
                log.warning(
                    'trafilatura_js_fallback failed; keeping trafilatura stubs: %s',
                    e,
                )
        elif needs_js_urls and not self.playwright_ws_url:
            log.debug(
                'trafilatura: %d URL(s) flagged needs_js but PLAYWRIGHT_WS_URL is unset; '
                'keeping trafilatura stubs',
                len(needs_js_urls),
            )

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
    query: Optional[str] = None,
    heading_trim_enabled: bool = True,
    heading_trim_min_token_len: int = 3,
    heading_trim_keep_intro: bool = True,
    js_fallback_enabled: bool = True,
    js_fallback_min_extract_chars: int = 200,
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
        # Heading-trim plumbing only flows through the trafilatura loader.
        # The Playwright loader doesn't emit markdown by default (its
        # default evaluator returns whatever ``page.evaluate`` produces);
        # if we ever swap that for a markdown-emitting evaluator the
        # trim can be reused here.
        if query:
            web_loader_args['query'] = query
        web_loader_args['heading_trim_enabled'] = heading_trim_enabled
        web_loader_args['heading_trim_min_token_len'] = heading_trim_min_token_len
        web_loader_args['heading_trim_keep_intro'] = heading_trim_keep_intro
        # JS-shell fallback: pass the Playwright sidecar endpoint and
        # timeout through so SafeTrafilaturaLoader.alazy_load can spin up
        # SafePlaywrightURLLoader for just the URLs that came back empty.
        # When PLAYWRIGHT_WS_URL is unset the fallback short-circuits
        # internally regardless of js_fallback_enabled.
        web_loader_args['js_fallback_enabled'] = js_fallback_enabled
        web_loader_args['js_fallback_min_extract_chars'] = js_fallback_min_extract_chars
        if PLAYWRIGHT_WS_URL.value:
            web_loader_args['playwright_ws_url'] = PLAYWRIGHT_WS_URL.value
        if PLAYWRIGHT_TIMEOUT.value:
            web_loader_args['playwright_timeout'] = PLAYWRIGHT_TIMEOUT.value

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
