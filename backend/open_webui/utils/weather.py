"""Current-weather lookup for the ``{{CURRENT_WEATHER}}`` prompt variable.

Pipeline (each layer is optional and fails open to ``''``):

1. **Parse the user's location string** (``user.info.location``).
   - If it parses as ``<lat>,<lon>`` → use directly.
   - Otherwise → split on the first comma, geocode the leading name via
     Open-Meteo, optionally disambiguate by matching the trailing region
     hint (US state abbreviation, full state/country name, ISO country
     code) against the geocoder's ``admin1`` / ``country_code`` fields.
2. **Fetch current conditions**:
   - **NWS / api.weather.gov** (US-only, no API key). Walks the
     ``/points/{lat,lon}`` -> ``/gridpoints/.../stations`` chain to find
     the nearest METAR-class station, then reads its latest observation.
     Skipped on 404 (outside US) or any error.
   - **Open-Meteo / api.open-meteo.com** fallback (global, no API key).
     Used for non-US lat/lon or when NWS fails.
3. **Format**: a single line like
   ``66°F, partly cloudy, wind 4 mph WSW, 71% humidity (Boston, MA)``.

Caching:

- Weather results: keyed by ``(location_str, units)``, TTL configurable
  via ``WEATHER_CACHE_TTL_SECONDS`` (default 600s = 10 min). Weather
  rarely changes intra-minute and an LLM prompt-injection variable
  doesn't need second-by-second precision.
- Geocoding results: keyed by ``location_str``, TTL one week. Place
  names don't move.

No external dependencies beyond ``requests`` (already in the project).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- config

_DEFAULT_UNITS = (os.getenv('WEATHER_UNITS', 'imperial') or 'imperial').lower()
if _DEFAULT_UNITS not in ('imperial', 'metric'):
    _DEFAULT_UNITS = 'imperial'

_WEATHER_TTL = max(60, int(os.getenv('WEATHER_CACHE_TTL_SECONDS', '600') or 600))
_GEO_TTL = 7 * 24 * 3600  # one week

_REQUEST_TIMEOUT = float(os.getenv('WEATHER_HTTP_TIMEOUT', '4') or 4)
_NWS_USER_AGENT = os.getenv(
    'WEATHER_USER_AGENT',
    'open-webui/weather (+https://openwebui.com)',
)

_NWS_BASE = 'https://api.weather.gov'
_OM_FORECAST = 'https://api.open-meteo.com/v1/forecast'
_OM_GEOCODE = 'https://geocoding-api.open-meteo.com/v1/search'

# -------------------------------------------------------------- HTTP sessions


def _build_session(default_ua: str) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({'GET'}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    s.headers.update({'User-Agent': default_ua})
    return s


_nws_session = _build_session(_NWS_USER_AGENT)
_om_session = _build_session('open-webui/weather (+https://openwebui.com)')


# ---------------------------------------------------------- in-process cache

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str, ttl: int) -> Optional[object]:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > ttl:
            _CACHE.pop(key, None)
            return None
        return value


def _cache_put(key: str, value: object) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


# --------------------------------------------------------- location parsing

# Direct ``lat,lon`` form. Accept signed floats; tolerate whitespace, and
# tolerate trailing text. The frontend's "Allow User Location" toggle
# (Settings -> Interface) writes the location as ``"42.360, -71.058 (lat,
# long)"`` -- the leading numbers are still the answer; we just ignore
# the explanatory suffix. The trailing ``\b`` (word boundary after the
# longitude) is what makes the suffix optional.
_LATLON_RE = re.compile(
    r'^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\b'
)


def _parse_latlon(s: str) -> Optional[tuple[float, float]]:
    m = _LATLON_RE.match(s)
    if not m:
        return None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (lat, lon)


# Two-letter US state abbreviations -> full name. Used to disambiguate
# geocoder hits when the user wrote ``Boston, MA`` -- the geocoder's
# ``admin1`` field returns the full state name (``Massachusetts``).
_US_STATE_ABBREV: dict[str, str] = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
}


def _score_geocode_hit(hit: dict, region_hint: Optional[str]) -> int:
    """Score a geocoder hit against the user's trailing region hint.

    Higher is better. Ties broken by Open-Meteo's own ordering (which is
    roughly population-weighted, so falling through to the first result
    when there's no signal is reasonable).
    """
    if not region_hint:
        return 0
    hint = region_hint.strip().lower()
    if not hint:
        return 0
    admin1 = (hit.get('admin1') or '').lower()
    country_code = (hit.get('country_code') or '').lower()
    country = (hit.get('country') or '').lower()

    # 2-letter US state hint -> map to full name and compare
    if len(hint) == 2 and hint.upper() in _US_STATE_ABBREV:
        full = _US_STATE_ABBREV[hint.upper()].lower()
        if admin1 == full:
            return 3
    if admin1 == hint or admin1.startswith(hint + ' '):
        return 3
    if country_code == hint or country == hint:
        return 2
    if hint in admin1 or hint in country:
        return 1
    return 0


def _geocode(location: str) -> Optional[tuple[float, float, str]]:
    """Resolve ``location`` to ``(lat, lon, display_name)``.

    ``display_name`` is the geocoder's canonical name (``Boston,
    Massachusetts``), used in the formatted output so the model knows
    *where* the weather is from when the user typed something terse.
    """
    cached = _cache_get(f'geo:{location.lower()}', _GEO_TTL)
    if cached is not None:
        return cached  # type: ignore[return-value]

    name_part, _, region_hint = location.partition(',')
    name_part = name_part.strip()
    region_hint = region_hint.strip() or None
    if not name_part:
        return None

    try:
        response = _om_session.get(
            _OM_GEOCODE,
            params={'name': name_part, 'count': 5, 'language': 'en'},
            timeout=_REQUEST_TIMEOUT,
        )
        if not response.ok:
            return None
        data = response.json() or {}
    except Exception as e:
        log.debug('weather: geocode failed for %r: %s', location, e)
        return None

    results = data.get('results') or []
    if not results:
        # Retry once with the second comma-split component as the search
        # term -- handles things like "Brooklyn, New York" where the
        # geocoder prefers "New York" as the name and "Brooklyn" as the
        # neighborhood.
        if region_hint:
            try:
                response = _om_session.get(
                    _OM_GEOCODE,
                    params={'name': region_hint, 'count': 3, 'language': 'en'},
                    timeout=_REQUEST_TIMEOUT,
                )
                if response.ok:
                    data = response.json() or {}
                    results = data.get('results') or []
            except Exception:
                pass
    if not results:
        return None

    # Best result by region-hint score, ties broken by source order
    # (Open-Meteo returns most-populous first for ambiguous names).
    scored = [
        (i, _score_geocode_hit(r, region_hint), r) for i, r in enumerate(results)
    ]
    scored.sort(key=lambda t: (-t[1], t[0]))
    best = scored[0][2]

    lat = best.get('latitude')
    lon = best.get('longitude')
    if lat is None or lon is None:
        return None

    name = best.get('name')
    admin1 = (best.get('admin1') or '').strip()
    cc = best.get('country_code')
    # Trim noisy prefixes that Open-Meteo includes on some admin1 names
    # (``State of Berlin``, ``Province of Quebec``, ...).
    for prefix in ('State of ', 'Province of ', 'Department of ', 'Region of '):
        if admin1.startswith(prefix):
            admin1 = admin1[len(prefix):]
            break
    display_parts: list[str] = [name] if name else []
    # Skip admin1 when it duplicates the city name (e.g. Tokyo/Tokyo,
    # Singapore/Singapore -- city-states and Japanese prefectures).
    if admin1 and admin1.lower() != (name or '').lower():
        if cc == 'US':
            inv = {v: k for k, v in _US_STATE_ABBREV.items()}
            display_parts.append(inv.get(admin1, admin1))
        else:
            display_parts.append(admin1)
    elif cc and not admin1:
        display_parts.append(cc)
    display = ', '.join(p for p in display_parts if p)

    result = (float(lat), float(lon), display)
    _cache_put(f'geo:{location.lower()}', result)
    return result


# --------------------------------------------------------------- NWS fetch


def _km_h_to_mph(kmh: float) -> float:
    return kmh * 0.621371


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _fetch_nws(lat: float, lon: float, units: str) -> Optional[dict]:
    """Return a flattened conditions dict from NWS, or None on failure.

    NWS only covers US territories. ``/points/{lat,lon}`` returns 404
    outside that region -- the caller (``get_current_weather``) treats
    that as "fall through to Open-Meteo".
    """
    try:
        r = _nws_session.get(
            f'{_NWS_BASE}/points/{lat:.4f},{lon:.4f}', timeout=_REQUEST_TIMEOUT
        )
        if r.status_code == 404:
            log.debug('weather: NWS points 404 for lat=%s lon=%s (outside US)', lat, lon)
            return None
        if not r.ok:
            log.debug('weather: NWS points HTTP %s', r.status_code)
            return None
        points = r.json() or {}
    except Exception as e:
        log.debug('weather: NWS points failed: %s', e)
        return None

    props = points.get('properties') or {}
    stations_url = props.get('observationStations')
    if not stations_url:
        return None

    try:
        r = _nws_session.get(stations_url, timeout=_REQUEST_TIMEOUT)
        if not r.ok:
            return None
        stations_doc = r.json() or {}
    except Exception as e:
        log.debug('weather: NWS stations failed: %s', e)
        return None

    features = stations_doc.get('features') or []
    if not features:
        return None

    # First station is the closest. If its observation is stale or null
    # we try the next two as fallbacks.
    station_ids = []
    for feat in features[:3]:
        sid = (feat.get('properties') or {}).get('stationIdentifier')
        if sid:
            station_ids.append(sid)

    for sid in station_ids:
        try:
            r = _nws_session.get(
                f'{_NWS_BASE}/stations/{sid}/observations/latest',
                timeout=_REQUEST_TIMEOUT,
            )
            if not r.ok:
                continue
            obs = (r.json() or {}).get('properties') or {}
        except Exception:
            continue

        def _val(field: str) -> Optional[float]:
            v = obs.get(field)
            if isinstance(v, dict):
                value = v.get('value')
                if isinstance(value, (int, float)):
                    return float(value)
            return None

        temp_c = _val('temperature')
        if temp_c is None:
            continue

        wind_kmh = _val('windSpeed')
        wind_dir = _val('windDirection')
        humidity = _val('relativeHumidity')
        text = obs.get('textDescription') or ''

        if units == 'imperial':
            temp = _c_to_f(temp_c)
            wind = _km_h_to_mph(wind_kmh) if wind_kmh is not None else None
            temp_unit = 'F'
            speed_unit = 'mph'
        else:
            temp = temp_c
            wind = wind_kmh
            temp_unit = 'C'
            speed_unit = 'km/h'

        # Pull the relative city from the original /points response so we
        # can name the location even when the user gave us raw lat/lon.
        rel = (props.get('relativeLocation') or {}).get('properties') or {}
        place = None
        if rel.get('city') and rel.get('state'):
            place = f"{rel['city']}, {rel['state']}"

        return {
            'temperature': temp,
            'temperature_unit': temp_unit,
            'conditions': text,
            'wind_speed': wind,
            'wind_speed_unit': speed_unit,
            'wind_direction': wind_dir,
            'humidity': humidity,
            'place': place,
            'source': f'NWS/{sid}',
        }

    return None


# ----------------------------------------------------------- Open-Meteo fetch

# Mapping from WMO weather code -> short text label. Mirrors the table
# Open-Meteo documents (https://open-meteo.com/en/docs#weathervariables).
# We collapse the day/night distinctions for brevity since the prompt
# already carries timestamps elsewhere.
_WMO_CODES: dict[int, str] = {
    0: 'Clear',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Depositing rime fog',
    51: 'Light drizzle',
    53: 'Drizzle',
    55: 'Dense drizzle',
    56: 'Light freezing drizzle',
    57: 'Freezing drizzle',
    61: 'Light rain',
    63: 'Rain',
    65: 'Heavy rain',
    66: 'Light freezing rain',
    67: 'Freezing rain',
    71: 'Light snow',
    73: 'Snow',
    75: 'Heavy snow',
    77: 'Snow grains',
    80: 'Light rain showers',
    81: 'Rain showers',
    82: 'Violent rain showers',
    85: 'Light snow showers',
    86: 'Snow showers',
    95: 'Thunderstorm',
    96: 'Thunderstorm with light hail',
    99: 'Thunderstorm with hail',
}


def _fetch_open_meteo(lat: float, lon: float, units: str) -> Optional[dict]:
    params: dict[str, str] = {
        'latitude': f'{lat:.4f}',
        'longitude': f'{lon:.4f}',
        'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m',
        'timezone': 'auto',
    }
    if units == 'imperial':
        params['temperature_unit'] = 'fahrenheit'
        params['wind_speed_unit'] = 'mph'
        params['precipitation_unit'] = 'inch'
        temp_unit = 'F'
        speed_unit = 'mph'
    else:
        params['temperature_unit'] = 'celsius'
        params['wind_speed_unit'] = 'kmh'
        temp_unit = 'C'
        speed_unit = 'km/h'

    try:
        r = _om_session.get(_OM_FORECAST, params=params, timeout=_REQUEST_TIMEOUT)
        if not r.ok:
            log.debug('weather: open-meteo HTTP %s', r.status_code)
            return None
        payload = r.json() or {}
    except Exception as e:
        log.debug('weather: open-meteo failed: %s', e)
        return None

    cur = payload.get('current') or {}
    temp = cur.get('temperature_2m')
    if not isinstance(temp, (int, float)):
        return None

    code = cur.get('weather_code')
    conditions = ''
    if isinstance(code, (int, float)):
        conditions = _WMO_CODES.get(int(code), '')

    return {
        'temperature': float(temp),
        'temperature_unit': temp_unit,
        'conditions': conditions,
        'wind_speed': cur.get('wind_speed_10m'),
        'wind_speed_unit': speed_unit,
        'wind_direction': cur.get('wind_direction_10m'),
        'humidity': cur.get('relative_humidity_2m'),
        'place': None,
        'source': 'Open-Meteo',
    }


# -------------------------------------------------------- formatting

_COMPASS_16 = (
    'N', 'NNE', 'NE', 'ENE',
    'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW',
    'W', 'WNW', 'NW', 'NNW',
)


def _bearing_to_compass(degrees: Optional[float]) -> str:
    if degrees is None:
        return ''
    try:
        d = float(degrees) % 360.0
    except (TypeError, ValueError):
        return ''
    idx = int((d + 11.25) // 22.5) % 16
    return _COMPASS_16[idx]


def _format(conditions_dict: dict, place_override: Optional[str]) -> str:
    temp = conditions_dict.get('temperature')
    temp_unit = conditions_dict.get('temperature_unit') or 'F'
    conditions = (conditions_dict.get('conditions') or '').strip()
    wind = conditions_dict.get('wind_speed')
    speed_unit = conditions_dict.get('wind_speed_unit') or 'mph'
    wind_dir = conditions_dict.get('wind_direction')
    humidity = conditions_dict.get('humidity')
    place = place_override or conditions_dict.get('place')

    parts: list[str] = []
    if isinstance(temp, (int, float)):
        parts.append(f'{round(float(temp))}°{temp_unit}')
    if conditions:
        # Fully lowercase so the joined line reads naturally:
        # "66°F, partly cloudy, ..." (NWS returns title case;
        # Open-Meteo's WMO-code mapping is mixed).
        parts.append(conditions.lower())
    if isinstance(wind, (int, float)):
        compass = _bearing_to_compass(wind_dir)
        wind_str = f'wind {round(float(wind))} {speed_unit}'
        if compass:
            wind_str += f' {compass}'
        parts.append(wind_str)
    if isinstance(humidity, (int, float)):
        parts.append(f'{round(float(humidity))}% humidity')

    summary = ', '.join(parts)
    if place:
        return f'{summary} ({place})' if summary else ''
    return summary


# -------------------------------------------------------- public entry point


def get_current_weather(
    location: Optional[str],
    units: Optional[str] = None,
) -> str:
    """Return a one-line weather summary for ``location``, or ``''``.

    Designed to be called inline from ``prompt_template`` (under
    ``asyncio.to_thread`` so we don't block the event loop). Every layer
    fails open: bad location, geocode miss, both APIs down -- the
    returned string is empty and the caller just substitutes nothing.

    Args:
        location: Free-text from ``user.info.location``. Can be
            ``lat,lon``, a city name, or ``City, Region``. Falsy /
            ``"None"`` / ``"unknown"`` short-circuits to ``''``.
        units: ``'imperial'`` or ``'metric'``. ``None`` uses the
            ``WEATHER_UNITS`` env default.
    """
    if not isinstance(location, str):
        return ''
    loc = location.strip()
    if not loc or loc.lower() in {'none', 'unknown', 'n/a', ''}:
        return ''

    units = (units or _DEFAULT_UNITS).lower()
    if units not in ('imperial', 'metric'):
        units = _DEFAULT_UNITS

    cache_key = f'wx:{loc.lower()}:{units}'
    cached = _cache_get(cache_key, _WEATHER_TTL)
    if isinstance(cached, str):
        return cached

    # Step 1: resolve to lat/lon (+ optional display name).
    place_override: Optional[str] = None
    parsed = _parse_latlon(loc)
    if parsed is not None:
        lat, lon = parsed
    else:
        geo = _geocode(loc)
        if geo is None:
            log.debug('weather: could not resolve location %r', loc)
            _cache_put(cache_key, '')  # negative cache prevents tight retry loops
            return ''
        lat, lon, display = geo
        place_override = display

    # Step 2: NWS primary, Open-Meteo fallback.
    conditions = _fetch_nws(lat, lon, units)
    if conditions is None:
        conditions = _fetch_open_meteo(lat, lon, units)
    if conditions is None:
        log.debug('weather: both providers failed for %r (lat=%s lon=%s)', loc, lat, lon)
        _cache_put(cache_key, '')
        return ''

    summary = _format(conditions, place_override)
    _cache_put(cache_key, summary)
    log.debug('weather: %r -> %r (source=%s)', loc, summary, conditions.get('source'))
    return summary


# ===================================================================
# Weather-intent short-circuit
# ===================================================================
# Mirrors ``nl_filter.is_pure_datetime_query`` / ``synthesize_datetime_answer``:
# when the user's question is *only* asking about current weather, we short-
# circuit the web-search pipeline by synthesizing an answer from NWS /
# Open-Meteo and injecting it as a synthetic "search result" file. No Kagi
# call, no LLM query expansion, no playwright fetch. The model sees the
# weather string as a regular source and answers naturally.


# Conservative phrasings only -- anything ambiguous ("nice outside?",
# "should I bring an umbrella?") deliberately falls through to real
# search. The grammar is whole-query, anchored, lowercased; smart quotes
# normalized before matching so iOS/macOS apostrophe-replacement doesn't
# break a match.
#
# Reusable optional tail handles the natural "outside / today / now /
# right now" appendage people stick on the end of weather questions.
_WX_TAIL = r"(?:\s+(?:outside|today|now|right\s+now|currently))?"

_WEATHER_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # weather (general)
    re.compile(
        r"^(?:what(?:'?s| is|s)?\s+|how(?:'?s| is|s)?\s+)?"
        r"(?:the\s+)?(?:current\s+|today'?s?\s+)?weather"
        r"(?:\s+like)?" + _WX_TAIL + r"\??$"
    ),
    re.compile(r"^how(?:'?s| is|s)?\s+(?:it|the\s+weather)" + _WX_TAIL + r"\??$"),
    # temperature
    re.compile(
        r"^(?:what(?:'?s| is|s)?\s+)?"
        r"(?:the\s+)?(?:current\s+|today'?s?\s+)?temp(?:erature)?"
        + _WX_TAIL + r"\??$"
    ),
    re.compile(
        r"^how\s+(?:hot|cold|warm|chilly|cool|freezing|humid|muggy)\s+is\s+it"
        + _WX_TAIL + r"\??$"
    ),
    # precipitation / sky
    re.compile(
        r"^is\s+it\s+(?:raining|snowing|sunny|cloudy|windy|humid|hailing|sleeting|foggy|stormy)"
        + _WX_TAIL + r"\??$"
    ),
    # humidity / wind
    re.compile(
        r"^(?:what(?:'?s| is|s)?\s+)?(?:the\s+)?(?:current\s+)?humidity"
        + _WX_TAIL + r"\??$"
    ),
    re.compile(
        r"^(?:what(?:'?s| is|s)?\s+)?(?:the\s+)?(?:current\s+)?wind(?:\s+speed)?"
        + _WX_TAIL + r"\??$"
    ),
)


_SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        '\u2018': "'",
        '\u2019': "'",
        '\u201B': "'",
        '\u201C': '"',
        '\u201D': '"',
        '\u201E': '"',
        '\u201F': '"',
    }
)


# Strips an optional ``in/at/near/around/for <place>`` segment that may be
# followed by another optional time qualifier ("right now", "today",
# etc.) before end-of-string. Matches both orderings the user is likely
# to type:
#
#   "weather in Athens right now?"   -> place = "athens",  time = "right now"
#   "weather right now in Athens?"   -> place = "athens",  time = None (the
#                                        "right now" was already absorbed
#                                        by _WX_TAIL on the inner pattern)
#   "weather in Boston, MA?"         -> place = "boston, ma"
#   "weather in New York City"       -> place = "new york city"
#
# Lazy ``+?`` plus the trailing ``\s*\??\s*$`` anchor means the engine
# extends the place greedily up to (but not into) an optional trailing
# time qualifier or the end of the string, which is what we want for
# multi-word place names like "san francisco bay area".
_WX_LOCATION_TAIL_RE = re.compile(
    r"\s+(?:in|at|near|around|for)\s+(?P<loc>.+?)"
    r"(?:\s+(?:outside|today|now|right\s+now|currently))?"
    r"\s*\??\s*$",
    re.IGNORECASE,
)


def _split_location_tail(q: str) -> tuple[str, Optional[str]]:
    """Return ``(stripped_query, location_hint)``.

    If ``q`` ends with an ``in <place>`` clause (with an optional
    trailing time qualifier), return the query with that clause removed
    and the extracted place name. Otherwise return ``(q, None)``.

    The returned query is reformatted so it can be re-matched against
    the location-free :data:`_WEATHER_QUERY_PATTERNS`: we strip the tail
    text in place and reattach a trailing ``?`` if the original had one,
    so e.g. ``"weather in Athens?"`` -> ``"weather?"``.
    """
    if not q:
        return q, None
    m = _WX_LOCATION_TAIL_RE.search(q)
    if not m:
        return q, None
    loc = (m.group('loc') or '').strip().rstrip(',.;:')
    if not loc:
        return q, None
    head = q[: m.start()].rstrip()
    if not head:
        # Defensive: don't strip the entire query down to nothing
        # ("in athens?" alone is not a weather query and should fall
        # through to real search).
        return q, None
    if q.rstrip().endswith('?') and not head.endswith('?'):
        head = head + '?'
    return head, loc


def is_weather_query(query: Optional[str]) -> bool:
    """True iff ``query`` is *only* asking about current weather.

    Whole-query match against a conservative pattern set. "What is the
    current temperature?" -> True; "Apple Q4 weather impact" -> False.
    Smart quotes are normalized first so the same regex covers
    iOS/macOS auto-replaced apostrophes.

    Also matches the same pattern set after stripping an optional
    ``in <place>`` tail, so "what is the weather in Boston?" and "is it
    raining in Seattle right now?" both return True. The extracted
    place is available via :func:`extract_weather_location_hint`.

    The match is intentionally narrow: anything with additional content
    words beyond the recognized intents + optional location/time tail
    (future-tense forecasts, weather-adjacent ambient questions like
    "should I bring an umbrella?") falls through to real search.
    """
    if not isinstance(query, str):
        return False
    q = query.strip().translate(_SMART_PUNCT_TRANSLATION).lower()
    if not q:
        return False
    if any(p.match(q) for p in _WEATHER_QUERY_PATTERNS):
        return True
    stripped, loc = _split_location_tail(q)
    if loc is None:
        return False
    return any(p.match(stripped) for p in _WEATHER_QUERY_PATTERNS)


def extract_weather_location_hint(query: Optional[str]) -> Optional[str]:
    """Return the ``in <place>`` location embedded in a weather query.

    Returns ``None`` when the query is not a weather query, has no
    location clause, or only the user's-profile-location form matches
    (e.g. "what's the temperature?" -> ``None``; "what's the
    temperature in Athens?" -> ``"athens"``).

    The returned string is lowercased and lightly trimmed but is
    otherwise raw user input; downstream ``get_current_weather`` parses
    it as either ``<lat>,<lon>`` or a free-text place name (geocoded
    via Open-Meteo with state/country disambiguation).
    """
    if not isinstance(query, str):
        return None
    q = query.strip().translate(_SMART_PUNCT_TRANSLATION).lower()
    if not q:
        return None
    # If the bare query already matches without a location tail, there's
    # no hint to extract -- caller should use the user's profile.
    if any(p.match(q) for p in _WEATHER_QUERY_PATTERNS):
        return None
    stripped, loc = _split_location_tail(q)
    if loc is None:
        return None
    if not any(p.match(stripped) for p in _WEATHER_QUERY_PATTERNS):
        return None
    return loc


def _extract_user_location(user: Any) -> Optional[str]:
    """Pull ``info.location`` off a user object (Pydantic model or dict).

    Mirrors the same shape ``prompt_template`` reads. Returns ``None``
    when the user has no location set, the value is falsy, or the
    sentinel ``"None"`` / ``"unknown"`` string -- so the caller can
    decide whether to short-circuit with a "no location" message or fall
    through to real search.
    """
    try:
        if hasattr(user, 'model_dump'):
            user = user.model_dump()
        if not isinstance(user, dict):
            return None
        info = user.get('info') or {}
        loc = info.get('location')
        if not isinstance(loc, str):
            return None
        loc = loc.strip()
        if not loc or loc.lower() in {'none', 'unknown', 'n/a'}:
            return None
        return loc
    except Exception:
        return None


def synthesize_weather_answer(
    user: Any,
    units: Optional[str] = None,
    location_override: Optional[str] = None,
) -> str:
    """Return a one-paragraph current-weather blurb for prompt injection.

    Used as the "page_content" of a synthetic web_search result so the
    downstream model treats it like any other source. Always returns a
    non-empty string -- callers don't need to handle empty results
    specially.

    When ``location_override`` is provided (typically extracted from
    the user's message via :func:`extract_weather_location_hint`), it
    takes precedence over the user's stored profile location. This is
    what routes queries like "weather in Athens?" to Athens rather
    than the asker's home town.

    When the override is absent and the user has no profile location
    set, returns a polite "no data" line that points them at the UI
    toggle they need to enable.
    """
    if location_override and isinstance(location_override, str):
        loc = location_override.strip()
        if loc:
            summary = get_current_weather(loc, units=units)
            if summary:
                return f'Current weather: {summary}.'
            return (
                f'Current weather lookup failed for {loc!r}. Neither NWS '
                '(api.weather.gov, US-only) nor Open-Meteo returned data. '
                'The location may be unrecognized or both upstream '
                'providers may be temporarily unavailable.'
            )

    location = _extract_user_location(user)
    if location is None:
        return (
            "Current weather is unavailable: the user has not set a location. "
            "They can enable it in Open WebUI under "
            "Settings -> Interface -> 'Allow User Location' (which uses the "
            "browser's geolocation API to write coordinates to their profile)."
        )

    summary = get_current_weather(location, units=units)
    if not summary:
        return (
            f'Current weather lookup failed for user location {location!r}. '
            'Neither NWS (api.weather.gov) nor Open-Meteo returned data. '
            'The location string may be unparseable, or both upstream '
            'providers may be temporarily unavailable.'
        )

    return f'Current weather: {summary}.'
