"""Taostats client.

The API key is read from the macOS Keychain at call time and kept in process
memory.  It is never passed on a command line, never written to the database,
never logged, and never included in an exception message -- every error path
here reports a status code and nothing else, because a failed request can echo
the request headers back and that is how keys leak into transcripts.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = "https://api.taostats.io/api"
KEYCHAIN_SERVICE = "taostats"
#: Taostats sits behind a WAF that answers a header-less request with 403
#: on EVERY path, including unauthenticated ones -- which reads exactly
#: like a rejected key and sent this debugging down the wrong road once.
#: A browser-shaped User-Agent is required before the key is even checked.
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
# Free tier allows 5 calls/minute.  Space requests at least this far apart so
# the ~5 back-to-back collector calls in one refresh can never burst past it.
MIN_REQUEST_INTERVAL = 20.0
_last_request_at = 0.0


class TaostatsError(RuntimeError):
    """Carries a status code only -- never a header, body or key."""


def _api_key(service: str = KEYCHAIN_SERVICE) -> str:
    try:
        done = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise TaostatsError(
            f"no Keychain item for service {service!r}") from None
    key = done.stdout.strip()
    if not key:
        raise TaostatsError(f"Keychain item {service!r} is empty")
    return key


def get(path: str, params: dict | None = None, *, key: str | None = None,
        timeout: int = 60, retries: int = 3):
    """GET a Taostats endpoint and return parsed JSON.

    ``key`` is passed in so a caller fetching many pages unlocks the Keychain
    once instead of once per request.
    """
    global _last_request_at
    key = key or _api_key()
    now = time.time()
    wait = MIN_REQUEST_INTERVAL - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{ROOT}/{path.lstrip('/')}{query}",
        headers={"Authorization": key, "accept": "application/json",
                 "User-Agent": USER_AGENT})
    last = 0
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            last = error.code              # deliberately not error.read()
            if error.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise TaostatsError(f"taostats {path} returned HTTP {last}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise TaostatsError(f"taostats {path} unreachable") from None
    raise TaostatsError(f"taostats {path} failed with HTTP {last}")
