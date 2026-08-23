import pytest

from civitapy.errors import (
    CivitAIAuthError,
    CivitAIBadRequestError,
    CivitAIDownloadError,
    CivitAIForbiddenError,
    CivitAIHTTPError,
    CivitAINotFoundError,
    CivitAIRateLimitError,
    CivitAIServerError,
    parse_error,
    raise_for_status,
)


def test_parse_error_none():
    assert parse_error(None) == ("Unknown error", None)


def test_parse_error_empty():
    assert parse_error({}) == ("Unknown error", None)


def test_parse_error_simple_format():
    assert parse_error({"error": "boom"}) == ("boom", None)


def test_parse_error_message_only():
    assert parse_error({"message": "oops"}) == ("oops", None)


def test_parse_error_code_without_message():
    assert parse_error({"code": "FORBIDDEN"}) == ("Error code FORBIDDEN", None)


def test_parse_error_message_and_issues():
    issues = [{"path": "name"}]
    assert parse_error({"message": "bad", "issues": issues}) == ("bad", issues)


def test_parse_error_prefers_message_over_code():
    assert parse_error({"code": "X", "message": "real"}) == ("real", None)


def test_raise_for_status_401():
    with pytest.raises(CivitAIAuthError):
        raise_for_status(401, {"error": "no auth"})


def test_raise_for_status_403():
    with pytest.raises(CivitAIForbiddenError):
        raise_for_status(403, {"error": "forbidden"})


def test_raise_for_status_404():
    with pytest.raises(CivitAINotFoundError):
        raise_for_status(404, {"error": "missing"})


def test_raise_for_status_400_with_issues():
    with pytest.raises(CivitAIBadRequestError) as exc:
        raise_for_status(400, {"message": "bad request", "issues": [{"path": "x"}]})
    assert exc.value.issues == [{"path": "x"}]


def test_raise_for_status_429():
    with pytest.raises(CivitAIRateLimitError) as exc:
        raise_for_status(429, {"error": "slow down"})
    assert exc.value.retry_after is None


def test_raise_for_status_500():
    with pytest.raises(CivitAIServerError) as exc:
        raise_for_status(500, {"error": "boom"})
    assert exc.value.status_code == 500


def test_raise_for_status_599():
    with pytest.raises(CivitAIServerError):
        raise_for_status(599, {})


def test_raise_for_status_other_4xx():
    with pytest.raises(CivitAIHTTPError) as exc:
        raise_for_status(418, {"error": "teapot"})
    assert exc.value.status_code == 418


def test_auth_error_details_appended():
    err = CivitAIAuthError("Unauthorized", "token missing")
    assert str(err) == "Unauthorized token missing"
    assert err.details == "token missing"


def test_auth_error_no_details():
    assert str(CivitAIAuthError()) == "Unauthorized"


def test_forbidden_error_details_appended():
    err = CivitAIForbiddenError("Forbidden", "gated file")
    assert str(err) == "Forbidden gated file"
    assert err.details == "gated file"


def test_bad_request_issues_default():
    assert CivitAIBadRequestError("x").issues == []


def test_server_error_message_format():
    err = CivitAIServerError(502, "gateway")
    assert "502" in str(err)
    assert "gateway" in str(err)
    assert err.status_code == 502


def test_http_error_message_format():
    err = CivitAIHTTPError(418, "teapot")
    assert str(err) == "HTTP 418: teapot"
    assert err.status_code == 418


def test_download_error_path():
    err = CivitAIDownloadError("failed", "/tmp/f")
    assert err.path == "/tmp/f"
    assert str(err) == "failed"


def test_rate_limit_retry_after():
    err = CivitAIRateLimitError("slow down", 5.0)
    assert err.retry_after == 5.0
