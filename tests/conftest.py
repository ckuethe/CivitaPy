import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from civitapy import CivitAIClient


class FakeResponse:
    """Minimal stand-in for an ``httpx.Response``."""

    def __init__(self, status_code=200, json_data=None, content=None, headers=None):
        self.status_code = status_code
        # ``_request`` only calls ``json()`` when content is truthy, so give JSON
        # responses a non-empty placeholder body.
        self.content = content if content is not None else (b"{}" if json_data is not None else b"")
        self.headers = headers or {}
        self._json_data = json_data
        self.is_success = 200 <= status_code < 300

    def json(self):
        return self._json_data

    async def aiter_bytes(self):
        if self.content:
            yield self.content


class FakeStream:
    """Async context manager yielding a :class:`FakeResponse` (``session.stream``)."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, responses):
        # Reference the shared pool (not a copy) so responses are consumed
        # across separate AsyncClient instances / requests.
        self._responses = responses
        self.request_calls = []

    def _next(self):
        if not self._responses:
            return FakeResponse()
        return self._responses.pop(0)

    async def request(self, method, url, **kwargs):
        self.request_calls.append((method, url, kwargs))
        return self._next()

    def stream(self, method, url, **kwargs):
        return FakeStream(self._next())


class FakeAsyncClient:
    """Async context manager used to replace ``httpx.AsyncClient``."""

    def __init__(self, responses=None, **kwargs):
        # Keep a reference to the shared pool; the pool is consumed globally.
        self._responses = responses if responses is not None else []

    async def __aenter__(self):
        return FakeSession(self._responses)

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def client(tmp_path):
    """A client with a throwaway download dir and no rate-limit spacing."""
    return CivitAIClient(
        base_url="https://example.com",
        download_dir=str(tmp_path),
        retry_count=3,
        min_request_interval=0.0,
    )


def mock_async_client(responses=None):
    """Context manager replacing ``httpx.AsyncClient`` with a canned fake.

    ``responses`` may be a single :class:`FakeResponse` or a list consumed in
    order across the request(s) made inside the patched block.
    """
    if responses is None:
        responses = [FakeResponse()]
    if isinstance(responses, FakeResponse):
        responses = [responses]
    pool = list(responses)

    def factory(*args, **kwargs):
        return FakeAsyncClient(pool)

    return mock.patch("civitapy.client.httpx.AsyncClient", factory)
