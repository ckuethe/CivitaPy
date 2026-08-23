import asyncio
from unittest import mock

import pytest

from civitapy.errors import (
    CivitAIAuthError,
    CivitAIBadRequestError,
    CivitAIForbiddenError,
    CivitAIHTTPError,
    CivitAINotFoundError,
    CivitAIRateLimitError,
    CivitAIServerError,
)

from conftest import FakeResponse, mock_async_client


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _no_sleep():
    with mock.patch("civitapy.client.asyncio.sleep", new=mock.AsyncMock()):
        yield


def test_request_success(client):
    resp = FakeResponse(200, {"items": [1]})
    with mock_async_client(resp):
        assert run(client.get("/models")) == {"items": [1]}


def test_request_204_returns_empty(client):
    resp = FakeResponse(204)
    with mock_async_client(resp):
        assert run(client.get("/nothing")) == {}


def test_request_trpc_unauthorized(client):
    resp = FakeResponse(401, {"code": "UNAUTHORIZED", "message": "bad token"})
    with mock_async_client(resp), pytest.raises(CivitAIAuthError):
        run(client.get("/me"))


def test_request_trpc_forbidden(client):
    resp = FakeResponse(403, {"code": "FORBIDDEN", "message": "no access"})
    with mock_async_client(resp), pytest.raises(CivitAIForbiddenError):
        run(client.get("/private"))


def test_request_trpc_other_code(client):
    resp = FakeResponse(400, {"code": "VALIDATION", "message": "bad", "issues": [{"path": "x"}]})
    with mock_async_client(resp), pytest.raises(CivitAIBadRequestError) as exc:
        run(client.get("/bad"))
    assert exc.value.issues == [{"path": "x"}]


def test_request_429_uses_retry_after_header(client):
    client._retry_count = 1  # disable retry so the 429 is surfaced
    resp = FakeResponse(429, {"error": "too many"}, headers={"Retry-After": "7"})
    with mock_async_client(resp), pytest.raises(CivitAIRateLimitError) as exc:
        run(client.get("/models"))
    assert exc.value.retry_after == 7.0


def test_request_generic_401(client):
    resp = FakeResponse(401, {"error": "no auth"})
    with mock_async_client(resp), pytest.raises(CivitAIAuthError):
        run(client.get("/me"))


def test_request_generic_404(client):
    resp = FakeResponse(404, {"error": "missing"})
    with mock_async_client(resp), pytest.raises(CivitAINotFoundError):
        run(client.get("/models/999"))


def test_request_generic_500(client):
    client._retry_count = 1  # disable retry so the 500 is surfaced
    resp = FakeResponse(500, {"error": "boom"})
    with mock_async_client(resp), pytest.raises(CivitAIServerError):
        run(client.get("/models"))


def test_request_generic_403_raises_http_error(client):
    # Civitai uses 403 as a payment-required signal; body isn't tRPC-shaped.
    resp = FakeResponse(403, {"error": "payment required"})
    with mock_async_client(resp), pytest.raises(CivitAIHTTPError) as exc:
        run(client.get("/models"))
    assert exc.value.status_code == 403


def test_request_other_4xx_raises_http_error(client):
    resp = FakeResponse(418, {"error": "teapot"})
    with mock_async_client(resp), pytest.raises(CivitAIHTTPError) as exc:
        run(client.get("/models"))
    assert exc.value.status_code == 418


def test_request_retries_429_then_succeeds(client):
    first = FakeResponse(429, {"error": "slow down"})
    second = FakeResponse(200, {"items": []})
    with mock_async_client([first, second]):
        assert run(client.get("/models")) == {"items": []}


def _fake_client_for(session_request):
    """Build an AsyncClient replacement whose session uses ``session_request``."""
    fake_session = mock.MagicMock()
    fake_session.request = session_request
    fake_client = mock.MagicMock()
    fake_client.__aenter__.return_value = fake_session
    fake_client.__aexit__.return_value = False
    return fake_client


def test_request_retries_http_error_then_succeeds(client):
    httpx = __import__("httpx")
    err = httpx.HTTPError("net fail")
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise err
        return FakeResponse(200, {"items": []})

    with mock.patch("civitapy.client.httpx.AsyncClient", return_value=_fake_client_for(flaky)):
        assert run(client.get("/models")) == {"items": []}
    assert calls["n"] == 2


def test_request_exhausts_http_errors(client):
    # Network failures on the final attempt are re-raised raw (not wrapped).
    httpx = __import__("httpx")
    err = httpx.HTTPError("net fail")

    async def always_fail(*args, **kwargs):
        raise err

    client._retry_count = 2
    with mock.patch("civitapy.client.httpx.AsyncClient", return_value=_fake_client_for(always_fail)):
        with pytest.raises(httpx.HTTPError):
            run(client._request("GET", "/models"))


def test_request_forwards_params(client):
    recorded = {}

    async def record(method, url, **kwargs):
        recorded["method"] = method
        recorded["kwargs"] = kwargs
        return FakeResponse(200, {"ok": True})

    with mock.patch("civitapy.client.httpx.AsyncClient", return_value=_fake_client_for(record)):
        run(client.post("/vault/toggle-version", params={"modelVersionId": 5}))
        assert recorded["kwargs"]["params"] == {"modelVersionId": 5}
        assert recorded["method"] == "POST"
