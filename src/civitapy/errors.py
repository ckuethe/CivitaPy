class CivitAIError(Exception):
    """Base exception for all CivitAI API errors."""


class CivitAIAuthError(CivitAIError):
    """Raised when authentication fails (401)."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message)


class CivitAIRateLimitError(CivitAIError):
    """Raised on 429 Too Many Requests."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message)


class CivitAINotFoundError(CivitAIError):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message)


class CivitAIBadRequestError(CivitAIError):
    """Raised on 400 Bad Request."""

    def __init__(self, message: str = "", issues: list[dict] | None = None):
        self.issues = issues or []
        super().__init__(message)


class CivitAIForbiddenError(CivitAIError):
    """Raised on 403 Forbidden."""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(message)


class CivitAIServerError(CivitAIError):
    """Raised on 5xx server errors."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"Server error ({status_code}): {message}" if message else f"Server error ({status_code})")


class CivitAIHTTPError(CivitAIError):
    """Raised for unexpected HTTP errors."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}" if message else f"HTTP {status_code}")


class CivitAIDownloadError(CivitAIError):
    """Raised when a downloaded file fails integrity verification.

    Indicates a size mismatch or, when the downloaded size matches the server's
    expected size, a SHA256 hash mismatch. The offending file is left in place so
    it can be inspected.
    """

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        super().__init__(message)


def parse_error(response_data: dict | None) -> tuple[str, list[dict] | None]:
    """Parse an error response body into (message, issues)."""
    if not response_data:
        return "Unknown error", None

    # Simple format: {"error": "..."}
    if "error" in response_data and isinstance(response_data["error"], str):
        return response_data["error"], None

    # Rich tRPC format: {"code": "...", "message": "...", "issues": [...]}
    message = ""
    issues = None
    if "message" in response_data:
        message = response_data["message"]
    if "code" in response_data and not message:
        message = f"Error code {response_data['code']}"
    if "issues" in response_data:
        issues = response_data["issues"]

    return message or "Unknown error", issues


def raise_for_status(status_code: int, data: dict | None):
    """Raise the appropriate CivitAI exception for an HTTP status code."""
    msg, issues = parse_error(data)

    if status_code == 401:
        raise CivitAIAuthError(msg)
    elif status_code == 403:
        raise CivitAIForbiddenError(msg)
    elif status_code == 404:
        raise CivitAINotFoundError(msg)
    elif status_code == 400:
        raise CivitAIBadRequestError(msg, issues)
    elif status_code == 429:
        raise CivitAIRateLimitError(msg)
    elif 500 <= status_code < 600:
        raise CivitAIServerError(status_code, msg)
    else:
        raise CivitAIHTTPError(status_code, msg)
