from __future__ import annotations

import asyncio
import hashlib
import logging
import os as _os_mod
from typing import Any, AsyncIterator, TypeVar

import httpx
from pydantic import BaseModel

from civitapy.errors import (
    CivitAIAuthError,
    CivitAIBadRequestError,
    CivitAIError,
    CivitAIHTTPError,
    CivitAINotFoundError,
    CivitAIRateLimitError,
    CivitAIServerError,
    CivitAIDownloadError,
    parse_error,
)
from civitapy.models import (
    Article,
    Collection,
    Creator,
    FullModelFile,
    HashLookupResult,
    Image,
    MiniHashes,
    MiniModelVersion,
    Model,
    ModelVersion,
    ModelVersionFile,
    TagItem,
    VaultInfo,
    VaultItem,
    VaultToggleResponse,
)

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)
_BASE_URL = "https://civitai.com/api/v1"


class _LoopGuard:
    """Manage event loop lifecycle for sync wrappers."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and not self._loop.is_closed():
            try:
                asyncio.get_running_loop()
                logger.warning(
                    "CivitAIClient sync methods should not be called from inside an async context. "
                    "Use the async API directly instead."
                )
                return self._loop
            except RuntimeError:
                pass

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            import atexit

            try:
                atexit.register(self._loop.close)
            except (RuntimeError, AttributeError):
                pass
        return self._loop

    def run(self, coro):
        loop = self.loop
        if hasattr(loop, "run_until_complete"):
            return loop.run_until_complete(coro)
        raise RuntimeError("Event loop is not usable")


_loop_guard = _LoopGuard()


def _parse_enum_list(value: str | list[str] | None) -> str | None:
    """Convert a single value or list into comma-separated string for API."""
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from params dict."""
    return {k: v for k, v in params.items() if v is not None}


class AsyncPaginator:
    """Async generator that paginates through API responses automatically."""

    def __init__(self, client: CivitAIClient, endpoint: str, model_cls: type[T], **params: Any):
        self._client = client
        self._endpoint = endpoint
        self._model_cls = model_cls
        self._params = params.copy()  # don't mutate caller's dict

    async def __aiter__(self) -> AsyncIterator[T]:
        has_cursor = "cursor" in self._params or not any(k.startswith("page") for k in self._params)

        if has_cursor:
            cursor = self._params.pop("cursor", None)
            while True:
                if cursor is not None:
                    self._params["cursor"] = cursor
                resp = await self._client._request("GET", self._endpoint, params=_clean_params(self._params))
                items = [self._model_cls(**item) for item in resp.get("items", [])]
                for item in items:
                    yield item

                next_cursor = (resp.get("metadata") or {}).get("nextCursor")
                if not next_cursor:
                    break
                cursor = next_cursor
        else:
            page = self._params.pop("page", 1)
            while True:
                self._params["page"] = page
                resp = await self._client._request("GET", self._endpoint, params=_clean_params(self._params))
                items = [self._model_cls(**item) for item in resp.get("items", [])]
                for item in items:
                    yield item

                metadata = resp.get("metadata") or {}
                next_page = metadata.get("nextPage") and page + 1
                total_pages = metadata.get("totalPages")
                if not next_page or (total_pages is not None and page >= total_pages):
                    break
                page += 1


class CivitAIClient:
    """Async client for the CivitAI Site API.

    Async usage::

        async with CivitAIClient() as client:
            models = await client.models_list(limit=50)
            model = await client.models_get(827184)

        # With auth token passed explicitly
        async with CivitAIClient(token="my-token") as client:
            user = await client.users_me()

    Sync usage (uses asyncio.run internally)::

        client = CivitAIClient()  # looks up CIVITAI_TOKEN env var automatically
        models = client.models_list(limit=50)

        # Explicit token overrides the environment variable
        client = CivitAIClient(token="my-token")
        user = client.users_me()
    """

    def __init__(self, *, token: str | None = None, base_url: str = _BASE_URL, timeout: float = 30.0):
        """Create a CivitAI API client.

        Args:
            token: Optional CivitAI API token. Defaults to the ``CIVITAI_TOKEN``
                environment variable when omitted. Required for authenticated
                endpoints (``/me``, vault, favorites/hidden filters, gated files).
            base_url: Base URL for the Site API. Defaults to the official endpoint.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        # Prefer explicit token arg → fall back to CIVITAI_TOKEN env var
        self._token = token or _os_mod.environ.get("CIVITAI_TOKEN")
        self._timeout = httpx.Timeout(timeout)

    @property
    def auth_header(self) -> dict[str, str] | None:
        """The ``Authorization: Bearer <token>`` header dict, or ``None`` when no token is set."""
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return None

    # -----------------------------------------------------------------------
    # Sync wrappers
    # -----------------------------------------------------------------------

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return _loop_guard.run(coro)

    # -- Enums --
    def enums(self) -> dict[str, list[str]]:
        """Fetch the current enum values used across the Site API (sync).

        Returns a dict of lists, e.g. ``ModelType``, ``ModelFileType``,
        ``ActiveBaseModel``, ``BaseModel``, ``BaseModelType``. Use these to
        discover valid values for filters like ``types=`` and ``baseModels=``
        rather than hardcoding them.

        See: ``GET /enums``
        """
        return self._run(self._enums_async())

    async def _enums_async(self) -> dict[str, list[str]]:
        """Fetch the current enum values used across the Site API.

        Returns only the list-valued keys (``ModelType``, ``BaseModel``, etc.),
        dropping scalar fields the response may include.
        """
        data = await self.get("/enums")
        return {k: v for k, v in data.items() if isinstance(v, list)}

    # -- Models --
    def models_list(
        self,
        *,
        limit: int = 100,
        page: int | None = None,
        cursor: str | None = None,
        query: str | None = None,
        ids: list[int] | None = None,
        tag: str | None = None,
        username: str | None = None,
        types: str | list[str] | None = None,
        base_models: str | list[str] | None = None,
        checkpoint_type: str | None = None,
        sort: str | None = "Highest Rated",
        period: str | None = "AllTime",
        nsfw: bool = False,
        supports_generation: bool | None = None,
        from_platform: bool | None = None,
        early_access: bool | None = None,
        primary_file_only: bool = False,
        favorites: bool | None = None,
        hidden: bool | None = None,
    ) -> dict[str, Any]:
        """List models with optional filters (sync wrapper for :meth:`models_list_async`).

        See :meth:`models_list_async` for the full parameter documentation.
        """
        return self._run(
            self.models_list_async(
                limit=limit,
                page=page,
                cursor=cursor,
                query=query,
                ids=ids,
                tag=tag,
                username=username,
                types=types,
                base_models=base_models,
                checkpoint_type=checkpoint_type,
                sort=sort,
                period=period,
                nsfw=nsfw,
                supports_generation=supports_generation,
                from_platform=from_platform,
                early_access=early_access,
                primary_file_only=primary_file_only,
                favorites=favorites,
                hidden=hidden,
            )
        )

    def models_get(self, id: int) -> dict[str, Any]:
        """Get a single model by ID (sync wrapper for :meth:`models_get_async`).

        See: ``GET /models/{id}``
        """
        return self._run(self.models_get_async(id))

    # -- Model Versions --
    def model_versions_get(self, id: int) -> dict[str, Any]:
        """Get a single model version by ID (sync wrapper for :meth:`model_versions_get_async`).

        See: ``GET /model-versions/{id}``
        """
        return self._run(self.model_versions_get_async(id))

    def model_versions_by_hash(self, hash_value: str) -> dict[str, Any] | None:
        """Look up a model version by file hash (sync wrapper for :meth:`model_versions_by_hash_async`).

        Accepts any hash type Civitai records (AutoV1/V2/V3, SHA256, BLAKE3, CRC32).
        See: ``GET /model-versions/by-hash/{hash}``
        """
        return self._run(self.model_versions_by_hash_async(hash_value))

    def model_versions_by_hash_bulk(self, hashes: list[str]) -> list[dict[str, Any]]:
        """Bulk-lookup model versions by up to 100 SHA256 hashes (sync wrapper for
        :meth:`model_versions_by_hash_bulk_async`).

        See: ``POST /model-versions/by-hash``
        """
        return self._run(self.model_versions_by_hash_bulk_async(hashes))

    def model_versions_by_hash_ids(self, hashes: list[str]) -> list[HashLookupResult]:
        """Resolve up to 10,000 SHA256 hashes to model version IDs (sync wrapper for
        :meth:`model_versions_by_hash_ids_async`).

        See: ``POST /model-versions/by-hash/ids``
        """
        return self._run(self.model_versions_by_hash_ids_async(hashes))

    def model_versions_mini(self, id: int, *, epoch: int | None = None) -> dict[str, Any]:
        """Get a minimal model version for downloading (sync wrapper for
        :meth:`model_versions_mini_async`).

        See: ``GET /model-versions/mini/{id}``
        """
        return self._run(self.model_versions_mini_async(id, epoch=epoch))

    # -- Images --
    def images_list(
        self,
        *,
        limit: int = 50,
        page: int | None = None,
        cursor: str | None = None,
        post_id: int | None = None,
        model_id: int | None = None,
        model_version_id: int | None = None,
        image_id: int | None = None,
        username: str | None = None,
        user_id: int | None = None,
        period: str | None = "AllTime",
        sort: str | None = "Most Reactions",
        nsfw: bool | str | None = None,
        browsing_level: int | None = None,
        tags: list[int] | None = None,
        type: str | None = None,
        base_models: str | list[str] | None = None,
        with_meta: bool = False,
    ) -> dict[str, Any]:
        """List images with optional filters (sync wrapper for :meth:`images_list_async`).

        See :meth:`images_list_async` for the full parameter documentation.
        """
        return self._run(
            self.images_list_async(
                limit=limit,
                page=page,
                cursor=cursor,
                post_id=post_id,
                model_id=model_id,
                model_version_id=model_version_id,
                image_id=image_id,
                username=username,
                user_id=user_id,
                period=period,
                sort=sort,
                nsfw=nsfw,
                browsing_level=browsing_level,
                tags=tags,
                type=type,
                base_models=base_models,
                with_meta=with_meta,
            )
        )

    # -- Articles --
    def articles_list(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        query: str | None = None,
        tags: list[int] | None = None,
        username: str | None = None,
        sort: str | None = "Newest",
        nsfw: bool = False,
    ) -> dict[str, Any]:
        """List articles with optional filters (sync wrapper for :meth:`articles_list_async`).

        See :meth:`articles_list_async` for the full parameter documentation.
        """
        return self._run(
            self.articles_list_async(
                limit=limit,
                cursor=cursor,
                query=query,
                tags=tags,
                username=username,
                sort=sort,
                nsfw=nsfw,
            )
        )

    def articles_get(self, id: int) -> dict[str, Any]:
        """Get a single article by ID (sync wrapper for :meth:`articles_get_async`).

        See: ``GET /articles/{id}``
        """
        return self._run(self.articles_get_async(id))

    # -- Collections --
    def collections_list(
        self,
        *,
        limit: int = 100,
        cursor: int | None = None,
        query: str | None = None,
        sort: str | None = "Newest",
        nsfw: bool = False,
    ) -> dict[str, Any]:
        """List public collections (sync wrapper for :meth:`collections_list_async`).

        See :meth:`collections_list_async` for the full parameter documentation.
        """
        return self._run(self.collections_list_async(limit=limit, cursor=cursor, query=query, sort=sort, nsfw=nsfw))

    def collections_get(self, id: int) -> dict[str, Any]:
        """Get a single public collection by ID (sync wrapper for :meth:`collections_get_async`).

        See: ``GET /collections/{id}``
        """
        return self._run(self.collections_get_async(id))

    # -- Creators --
    def creators_list(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        """List creators (sync wrapper for :meth:`creators_list_async`).

        See: ``GET /creators``
        """
        return self._run(self.creators_list_async(limit=limit, page=page, query=query))

    # -- Tags --
    def tags_list(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        """List model tags (sync wrapper for :meth:`tags_list_async`).

        See: ``GET /tags``
        """
        return self._run(self.tags_list_async(limit=limit, page=page, query=query))

    # -- Users --
    def users_me(self) -> dict[str, Any]:
        """Get the authenticated caller's profile (sync wrapper for :meth:`users_me_async`).

        Requires a valid token. See: ``GET /me``
        """
        return self._run(self.users_me_async())

    def users_lookup(
        self,
        *,
        ids: list[int] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Look up users by ID or username prefix (sync wrapper for :meth:`users_lookup_async`).

        See: ``GET /users``
        """
        return self._run(self.users_lookup_async(ids=ids, query=query))

    # -- Permissions --
    def permissions_check(
        self,
        entity_ids: list[int],
        *,
        entity_type: str = "ModelVersion",
        permission: str = "Generate",
        user_id: int | None = None,
    ) -> dict[str, bool]:
        """Check generation permission for model version IDs (sync wrapper for
        :meth:`permissions_check_async`).

        See: ``GET /permissions/check``
        """
        return self._run(
            self.permissions_check_async(entity_ids, entity_type=entity_type, permission=permission, user_id=user_id)
        )

    # -- Vault --
    def vault_get(self) -> dict[str, Any]:
        """Get (or create) the caller's vault (sync wrapper for :meth:`vault_get_async`).

        Requires an active membership and a valid token. See: ``GET /vault/get``
        """
        return self._run(self.vault_get_async())

    def vault_all(
        self,
        *,
        limit: int = 60,
        page: int = 1,
        query: str | None = None,
        types: list[str] | None = None,
        categories: list[str] | None = None,
        base_models: list[str] | None = None,
        date_created_from: str | None = None,
        date_created_to: str | None = None,
        date_added_from: str | None = None,
        date_added_to: str | None = None,
        sort: str | None = "Recently Added",
    ) -> dict[str, Any]:
        """List vault items with optional filters (sync wrapper for :meth:`vault_all_async`).

        See :meth:`vault_all_async` for the full parameter documentation.
        """
        return self._run(
            self.vault_all_async(
                limit=limit,
                page=page,
                query=query,
                types=types,
                categories=categories,
                base_models=base_models,
                date_created_from=date_created_from,
                date_created_to=date_created_to,
                date_added_from=date_added_from,
                date_added_to=date_added_to,
                sort=sort,
            )
        )

    def vault_check(self, model_version_ids: list[int]) -> list[dict[str, Any]]:
        """Check which model versions are in the caller's vault (sync wrapper for
        :meth:`vault_check_async`).

        See: ``GET /vault/check-vault``
        """
        return self._run(self.vault_check_async(model_version_ids))

    def vault_toggle(self, model_version_id: int) -> dict[str, Any]:
        """Add or remove a model version from the caller's vault (sync wrapper for
        :meth:`vault_toggle_async`).

        Idempotent — toggles membership. See: ``POST /vault/toggle-version``
        """
        return self._run(self.vault_toggle_async(model_version_id))

    # -- Downloads --
    def download_model_version(
        self,
        version_id: int,
        destination_dir: str,
        *,
        filename: str | None = None,
        base_model: str | None = None,
    ) -> list[str]:
        """Download every file of a single model version (sync wrapper for
        :meth:`download_model_version_async`).

        See :meth:`download_model_version_async` for the full parameter documentation.
        """
        return self._run(self.download_model_version_async(version_id, destination_dir, filename=filename, base_model=base_model))

    def download_model(
        self,
        model_id: int,
        destination_dir: str,
        *,
        base_model: str | None = None,
    ) -> list[str]:
        """Download all files of every version of a model (sync wrapper for
        :meth:`download_model_async`).

        See :meth:`download_model_async` for the full parameter documentation.
        """
        return self._run(self.download_model_async(model_id, destination_dir, base_model=base_model))

    # -----------------------------------------------------------------------
    # Core HTTP (async + sync wrappers delegate to these via _request)
    # -----------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | list | None = None,
        retry_count: int = 3,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = self.auth_header or {}
        last_exc = CivitAIError("No response received")

        for attempt in range(1, retry_count + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as session:
                    resp = await session.request(method, url, headers=headers, params=params, json=json)

                if resp.status_code == 204:
                    return {}

                data = resp.json() if resp.content else None
                error_data = data or {}
                is_trpc_error = "code" in error_data and "message" in error_data

                # tRPC-style errors (UNAUTHORIZED, FORBIDDEN, etc.) before generic check
                if is_trpc_error and resp.status_code < 500:
                    code = str(error_data["code"])
                    msg = error_data["message"]
                    issues = error_data.get("issues", [])

                    if code in ("UNAUTHORIZED",):
                        raise CivitAIAuthError(msg)
                    elif code == "FORBIDDEN":
                        from civitapy.errors import CivitAIForbiddenError

                        raise CivitAIForbiddenError(msg)
                    else:
                        raise CivitAIBadRequestError(msg, issues)

                # 429 before generic error handling (need Retry-After header access)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 1))
                    raise CivitAIRateLimitError(data.get("error") or data.get("message"), retry_after)

                # Generic error handling for non-success status codes
                if not resp.is_success:
                    msg, _ = parse_error(error_data)

                    if resp.status_code == 401:
                        raise CivitAIAuthError(msg)
                    elif resp.status_code == 404:
                        raise CivitAINotFoundError(msg)
                    elif resp.status_code >= 500:
                        raise CivitAIServerError(resp.status_code, msg)
                    else:
                        raise CivitAIHTTPError(resp.status_code, msg)

                return data or {}

            except (CivitAIRateLimitError, CivitAIServerError) as e:
                if attempt == retry_count:
                    raise
                delay = min(2**attempt * 0.5, 30)
                logger.warning("Request to %s failed (%s), retrying in %.1fs", path, type(e).__name__, delay)
                await asyncio.sleep(delay)

            except httpx.HTTPError as e:
                last_exc = CivitAIHTTPError(0, str(e))
                if attempt == retry_count:
                    raise
                await asyncio.sleep(min(2**attempt * 0.5, 30))

        raise last_exc

    # -----------------------------------------------------------------------
    # Async methods (the real implementations)
    # -----------------------------------------------------------------------

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a GET request to the Site API (low-level).

        Args:
            path: API path, e.g. ``/models`` (appended to the base URL).
            params: Optional query parameters.

        Returns:
            The parsed JSON response as a dict (``{}`` on 204).
        """
        return await self._request("GET", path, params=params)

    async def post(self, path: str, *, json: dict[str, Any] | list | None = None) -> dict[str, Any]:
        """Send a POST request to the Site API (low-level).

        Args:
            path: API path, e.g. ``/model-versions/by-hash``.
            json: Optional JSON request body.

        Returns:
            The parsed JSON response as a dict.
        """
        return await self._request("POST", path, json=json)

    async def _enums_async(self) -> dict[str, list[str]]:
        data = await self.get("/enums")
        return {k: v for k, v in data.items() if isinstance(v, list)}

    # -- Models (async) --
    async def models_list_async(
        self,
        *,
        limit: int = 100,
        page: int | None = None,
        cursor: str | None = None,
        query: str | None = None,
        ids: list[int] | None = None,
        tag: str | None = None,
        username: str | None = None,
        types: str | list[str] | None = None,
        base_models: str | list[str] | None = None,
        checkpoint_type: str | None = None,
        sort: str | None = "Highest Rated",
        period: str | None = "AllTime",
        nsfw: bool = False,
        supports_generation: bool | None = None,
        from_platform: bool | None = None,
        early_access: bool | None = None,
        primary_file_only: bool = False,
        favorites: bool | None = None,
        hidden: bool | None = None,
    ) -> dict[str, Any]:
        """List models with optional filters.

        Auth: Mixed — ``favorites`` and ``hidden`` require a bearer token.
        ``page * limit`` above 1000 returns 429; use ``cursor`` for deep paging.
        Combining ``query`` with ``page`` returns 400 (use cursor-based paging).

        Args:
            limit: Items per page (1–100, default 100).
            page: 1-indexed page number (incompatible with ``query``).
            cursor: Opaque pagination cursor from ``metadata.nextCursor``.
            query: Full-text search (Meilisearch); requires cursor pagination.
            ids: Restrict to specific model IDs.
            tag: Filter by tag name.
            username: Filter by creator (auto-slugified).
            types: One or more ``ModelType`` values.
            base_models: Base model(s) to filter by (e.g. ``SDXL 1.0``).
            checkpoint_type: For checkpoints only (``Standard``/``Trained``/``Merge``).
            sort: Sort order (see ``SortOrder``).
            period: Time window for sort metrics (see ``Period``).
            nsfw: Include mature content when ``True``.
            supports_generation: Only models supported by on-site generation.
            from_platform: Only models trained on Civitai.
            early_access: Include early-access versions.
            primary_file_only: Drop non-primary files from each version's ``files[]``.
            favorites: *(auth)* Only models in the caller's bookmarks.
            hidden: *(auth)* Only models the caller has hidden.
        """
        params = _clean_params(
            {
                "limit": limit if limit else 100,
                "page": page,
                "cursor": cursor,
                "query": query,
                "ids": ",".join(str(i) for i in ids) if ids else None,
                "tag": tag,
                "username": username,
                "types": _parse_enum_list(types),
                "baseModels": _parse_enum_list(base_models),
                "checkpointType": checkpoint_type,
                "sort": sort,
                "period": period,
                "nsfw": nsfw or None,
                "supportsGeneration": supports_generation,
                "fromPlatform": from_platform,
                "earlyAccess": early_access,
                "primaryFileOnly": primary_file_only if not limit else None,
                "favorites": favorites,
                "hidden": hidden,
            }
        )

        if page and params.get("limit"):
            effective_limit = int(params["limit"])
            if page * effective_limit > 1000:
                raise CivitAIRateLimitError("You've requested too many pages, please use cursors instead")

        return await self.get("/models", params=params)

    async def models_get_async(self, id: int) -> dict[str, Any]:
        """Get a single model by ID.

        Returns the same shape as one item from :meth:`models_list_async`.
        Raises :class:`CivitAINotFoundError` if the model doesn't exist.
        """
        return await self.get(f"/models/{id}")

    # -- Model Versions (async) --
    async def model_versions_get_async(self, id: int) -> dict[str, Any]:
        """Get a single model version by ID.

        Includes the full ``files[]`` list with hashes and download URLs. A
        valid token exposes extra early-access fields. Raises
        :class:`CivitAINotFoundError` if the version doesn't exist or isn't published.
        """
        return await self.get(f"/model-versions/{id}")

    async def model_versions_by_hash_async(self, hash_value: str) -> dict[str, Any] | None:
        """Look up a model version by a file hash.

        Accepts any hash type Civitai records (AutoV1/V2/V3, SHA256, BLAKE3, CRC32),
        matched case-insensitively. Returns ``None`` when no matching file is found
        (instead of raising).
        """
        try:
            data = await self.get(f"/model-versions/by-hash/{hash_value}")
            return data if isinstance(data, dict) else None
        except CivitAINotFoundError:
            return None

    async def model_versions_by_hash_bulk_async(self, hashes: list[str]) -> list[dict[str, Any]]:
        """Bulk-lookup model versions by up to 100 SHA256 hashes.

        Each hash must be a full 64-char SHA256; malformed or non-64-char hashes
        raise a bad-request error. Unmatched hashes are silently dropped.
        """
        if len(hashes) > 100:
            raise ValueError("Maximum 100 hashes per request")
        data = await self.post("/model-versions/by-hash", json=hashes)
        return data if isinstance(data, list) else []

    async def model_versions_by_hash_ids_async(self, hashes: list[str]) -> list[HashLookupResult]:
        """Resolve up to 10,000 SHA256 hashes to model version IDs only.

        Cheaper than :meth:`model_versions_by_hash_bulk_async` — returns
        ``{modelVersionId, hash}`` pairs. Unmatched hashes are silently dropped.
        """
        if len(hashes) > 10_000:
            raise ValueError("Maximum 10,000 hashes per request")
        data = await self.post("/model-versions/by-hash/ids", json=hashes)
        return [HashLookupResult(**item) for item in (data if isinstance(data, list) else [])]

    async def model_versions_mini_async(self, id: int, *, epoch: int | None = None) -> dict[str, Any]:
        """Get a minimal model version, trimmed for downloading.

        Skips heavy fields (``images[]``, ``description``, full ``files[]``) and
        returns just the download URLs, size, and hashes needed to fetch a file.
        ``epoch`` selects a specific epoch's file for private training results.
        """
        params = _clean_params({"epoch": epoch}) if epoch is not None else {}
        return await self.get(f"/model-versions/mini/{id}", params=params)

    # -- Images (async) --
    async def images_list_async(
        self,
        *,
        limit: int = 50,
        page: int | None = None,
        cursor: str | None = None,
        post_id: int | None = None,
        model_id: int | None = None,
        model_version_id: int | None = None,
        image_id: int | None = None,
        username: str | None = None,
        user_id: int | None = None,
        period: str | None = "AllTime",
        sort: str | None = "Most Reactions",
        nsfw: bool | str | None = None,
        browsing_level: int | None = None,
        tags: list[int] | None = None,
        type: str | None = None,
        base_models: str | list[str] | None = None,
        with_meta: bool = False,
    ) -> dict[str, Any]:
        """List images with optional filters.

        Authenticated callers see content up to their configured browsing level;
        anonymous callers are capped at the public browsing level. ``nsfw`` is a
        legacy filter — prefer ``browsing_level`` (an integer bitmask) which takes
        precedence. Page-based pagination is capped at ``page * limit <= 1000``.

        Args:
            limit: Items per page (0–200, default 50).
            page: 1-indexed page number (incompatible with ``cursor``).
            cursor: Opaque pagination cursor from ``metadata.nextCursor``.
            post_id: Restrict to a specific post.
            model_id: Images associated with any version of a model.
            model_version_id: Images associated with a specific version.
            image_id: Single-image lookup.
            username: Filter by uploader username.
            user_id: Filter by uploader user ID.
            period: Time window for sort metrics.
            sort: Sort order (see ``ImageSortOrder``).
            nsfw: Legacy NSFW filter (``None``/``Soft``/``Mature``/``X`` or bool).
            browsing_level: Raw browsing-level bitmask; overrides ``nsfw``.
            tags: Tag IDs required on each image.
            type: Media type (``image``/``video``/``audio``).
            base_models: Base models to restrict to.
            with_meta: Include the full ``meta`` object when ``True``.
        """
        params = _clean_params(
            {
                "limit": limit or 50,
                "page": page,
                "cursor": cursor,
                "postId": post_id,
                "modelId": model_id,
                "modelVersionId": model_version_id,
                "imageId": image_id,
                "username": username,
                "userId": user_id,
                "period": period,
                "sort": sort,
                "nsfw": nsfw if isinstance(nsfw, str) else (None if nsfw is False else nsfw),
                "browsingLevel": browsing_level,
                "tags": ",".join(str(t) for t in tags) if tags else None,
                "type": type,
                "baseModels": _parse_enum_list(base_models),
                "withMeta": with_meta or None,
            }
        )

        if page and params.get("limit"):
            effective_limit = int(params["limit"])
            if page * effective_limit > 1000:
                raise CivitAIRateLimitError("You've requested too many pages, please use cursors instead")

        return await self.get("/images", params=params)

    # -- Articles (async) --
    async def articles_list_async(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        query: str | None = None,
        tags: list[int] | None = None,
        username: str | None = None,
        sort: str | None = "Newest",
        nsfw: bool = False,
    ) -> dict[str, Any]:
        """List articles with optional filters.

        Public and edge-cached; only published, scanned, non-private articles are
        returned. Articles use cursor-based pagination only (no ``page`` param).

        Args:
            limit: Items per page (1–100, default 100).
            cursor: Opaque keyset cursor from ``metadata.nextCursor``.
            query: Full-text search over the article title.
            tags: Tag IDs (not names) to filter by.
            username: Filter by author username.
            sort: Sort order (see ``ArticleSortOrder``).
            nsfw: Include mature content when ``True`` (clamped to SFW in restricted regions).
        """
        params = _clean_params(
            {
                "limit": limit or 100,
                "cursor": cursor,
                "query": query,
                "tags": ",".join(str(t) for t in tags) if tags else None,
                "username": username,
                "sort": sort,
                "nsfw": nsfw or None,
            }
        )
        return await self.get("/articles", params=params)

    async def articles_get_async(self, id: int) -> dict[str, Any]:
        """Get a single article by ID, including the HTML body.

        Raises :class:`CivitAINotFoundError` if the article doesn't exist or is a
        draft/unpublished/private article (indistinguishable from missing).
        """
        return await self.get(f"/articles/{id}")

    # -- Collections (async) --
    async def collections_list_async(
        self,
        *,
        limit: int = 100,
        cursor: int | None = None,
        query: str | None = None,
        sort: str | None = "Newest",
        nsfw: bool = False,
    ) -> dict[str, Any]:
        """List public collections with optional filters.

        Only public collections are returned; private ones are indistinguishable
        from missing. ``cursor`` is only supported with the default ``Newest`` sort.

        Args:
            limit: Items per page (1–100, default 100).
            cursor: Keyset cursor (a collection ID) from ``metadata.nextCursor``.
            query: Full-text search over the collection name.
            sort: Sort order (``Newest`` or ``Most Followers``).
            nsfw: Include mature content when ``True``.
        """
        params = _clean_params(
            {"limit": limit or 100, "cursor": cursor, "query": query, "sort": sort, "nsfw": nsfw or None}
        )
        return await self.get("/collections", params=params)

    async def collections_get_async(self, id: int) -> dict[str, Any]:
        """Get a single public collection by ID.

        Raises :class:`CivitAINotFoundError` if the collection doesn't exist or is
        private (indistinguishable from missing).
        """
        return await self.get(f"/collections/{id}")

    # -- Creators (async) --
    async def creators_list_async(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        """List creators (users who have published at least one model).

        Page-based pagination only (no cursor). Sorted alphabetically by username;
        scope deep traversals with ``query=`` rather than linear paging.

        Args:
            limit: Items per page (1–200, default 20).
            page: 1-indexed page number.
            query: Full-text search on username.
        """
        params = _clean_params({"limit": limit or 20, "page": page, "query": query})

        if page * (params.get("limit") or 20) > 1000:
            raise CivitAIRateLimitError("You've requested too many pages, please use cursors instead")

        return await self.get("/creators", params=params)

    # -- Tags (async) --
    async def tags_list_async(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        """List model tags.

        Scoped to model tags (``entityType=Model``). ``totalItems``/``totalPages``
        may be reported as 0 — drive pagination off ``nextPage`` instead.

        Args:
            limit: Items per page (1–200, default 20).
            page: 1-indexed page number.
            query: Full-text search on tag name.
        """
        params = _clean_params({"limit": limit or 20, "page": page, "query": query})

        if page * (params.get("limit") or 20) > 1000:
            raise CivitAIRateLimitError("You've requested too many pages, please use cursors instead")

        return await self.get("/tags", params=params)

    # -- Users (async) --
    async def users_me_async(self) -> dict[str, Any]:
        """Get the authenticated caller's profile.

        Requires a valid token (returns 401 otherwise). Returns account info,
        membership tier, status, and subscription names.
        """
        return await self.get("/me")

    async def users_lookup_async(
        self,
        *,
        ids: list[int] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Look up users by ID or username prefix.

        Returns lean ``{id, username}`` results. Deleted/system users are filtered
        out. At least one of ``ids`` or ``query`` should be supplied.

        Args:
            ids: Specific user IDs to look up.
            query: Username prefix match.
        """
        params = _clean_params({"ids": ",".join(str(i) for i in ids) if ids else None, "query": query})
        return await self.get("/users", params=params)

    # -- Permissions (async) --
    async def permissions_check_async(
        self,
        entity_ids: list[int],
        *,
        entity_type: str = "ModelVersion",
        permission: str = "Generate",
        user_id: int | None = None,
    ) -> dict[str, bool]:
        """Check whether a user is allowed to generate from the given resources.

        Runs against the user identified by ``user_id`` (anonymous when omitted);
        bearer tokens are not used to scope this endpoint. Returns a flat mapping
        of each entity ID to a boolean.

        Args:
            entity_ids: The model version IDs to check (required).
            entity_type: Entity kind (only ``ModelVersion`` is supported).
            permission: Permission to check (only ``Generate`` is supported).
            user_id: Check on behalf of this user instead of the token's owner.
        """
        params = _clean_params(
            {
                "entityIds": ",".join(str(i) for i in entity_ids),
                "entityType": entity_type,
                "permission": permission,
                "userId": user_id,
            }
        )
        data = await self.get("/permissions/check", params=params)
        return {str(k): bool(v) for k, v in data.items()} if isinstance(data, dict) else {}

    # -- Vault (async) --
    async def vault_get_async(self) -> dict[str, Any]:
        """Get (or create) the caller's vault.

        Requires an active membership and a valid token. Free-tier callers get a
        ``vault: null`` response. See: ``GET /vault/get``
        """
        return await self.get("/vault/get")

    async def vault_all_async(
        self,
        *,
        limit: int = 60,
        page: int = 1,
        query: str | None = None,
        types: list[str] | None = None,
        categories: list[str] | None = None,
        base_models: list[str] | None = None,
        date_created_from: str | None = None,
        date_created_to: str | None = None,
        date_added_from: str | None = None,
        date_added_to: str | None = None,
        sort: str | None = "Recently Added",
    ) -> dict[str, Any]:
        """List items in the caller's vault with optional filters.

        Requires an active membership and a valid token. ``query`` does a
        case-insensitive substring match against model/version/creator names.

        Args:
            limit: Items per page (1–200, default 60).
            page: 1-indexed page number.
            query: Substring match against model/version/creator names.
            types: Model types to include (e.g. ``Checkpoint,LORA``).
            categories: Categories to filter by.
            base_models: Base models to filter by (e.g. ``SDXL 1.0,Flux.1 D``).
            date_created_from/to: Bound the model version's ``createdAt``.
            date_added_from/to: Bound when the item was added to the vault.
            sort: Sort order (see ``VaultSortOrder``).
        """
        params = _clean_params(
            {
                "limit": limit or 60,
                "page": page,
                "query": query,
                "types": ",".join(types) if types else None,
                "categories": ",".join(categories) if categories else None,
                "baseModels": _parse_enum_list(base_models),
                "dateCreatedFrom": date_created_from,
                "dateCreatedTo": date_created_to,
                "dateAddedFrom": date_added_from,
                "dateAddedTo": date_added_to,
                "sort": sort,
            }
        )

        if page * (params.get("limit") or 60) > 1000:
            raise CivitAIRateLimitError("You've requested too many pages, please use cursors instead")

        return await self.get("/vault/all", params=params)

    async def vault_check_async(self, model_version_ids: list[int]) -> list[dict[str, Any]]:
        """Check which model versions are in the caller's vault.

        Returns an array with one entry per requested ID; ``vaultItem`` is ``null``
        when the version isn't in the vault. See: ``GET /vault/check-vault``
        """
        params = _clean_params({"modelVersionIds": ",".join(str(i) for i in model_version_ids)})
        data = await self.get("/vault/check-vault", params=params)
        return data if isinstance(data, list) else []

    async def vault_toggle_async(self, model_version_id: int) -> dict[str, Any]:
        """Add or remove a model version from the caller's vault.

        Idempotent — toggles membership (adds if absent, removes if present).
        ``vaultId`` is omitted when the operation removed the item.
        See: ``POST /vault/toggle-version``
        """
        return await self.post("/vault/toggle-version", params={"modelVersionId": model_version_id})

    # -- Downloads (async) --

    async def download_model_version_async(
        self,
        version_id: int,
        destination_dir: str,
        *,
        filename: str | None = None,
        base_model: str | None = None,
    ) -> list[str]:
        """Download every file of a single model version.

        Files are written to ``destination_dir`` using their API names by default.
        When ``filename`` is given, it is used for the primary file (or the only
        file when there is just one); any remaining files keep their API names.
        Interrupted downloads are stored as ``<name>.part`` and resumed on a later
        call. Each file's size is verified against the API's ``sizeKB``; once the
        downloaded size is at least ``int(sizeKB * 1024)`` its SHA256 is also
        checked against the API's ``SHA256`` hash.

        Args:
            version_id: The model version ID to download.
            destination_dir: Directory to write files into (created if missing).
            filename: Optional custom filename for the primary file.
            base_model: If given and it doesn't match the version's base model,
                nothing is downloaded and an empty list is returned.

        Returns:
            A list of absolute paths to the successfully downloaded files.

        Raises:
            CivitAIDownloadError: If a file's size or SHA256 fails verification.
        """
        data = await self.model_versions_get_async(version_id)
        version = ModelVersion(**data)
        if base_model is not None and version.base_model != base_model:
            return []

        downloaded: list[str] = []
        for file in version.files:
            use_name = filename if (file.primary or len(version.files) == 1) else None
            path = await self._download_file_async(file, destination_dir, filename=use_name)
            if path:
                downloaded.append(path)
        return downloaded

    async def download_model_async(
        self,
        model_id: int,
        destination_dir: str,
        *,
        base_model: str | None = None,
    ) -> list[str]:
        """Download all files of every version of a model.

        Fetches the model and downloads each version's files into
        ``destination_dir``. When ``base_model`` is given, only versions whose
        base model matches are downloaded. Files use their API names.

        Args:
            model_id: The model ID to download.
            destination_dir: Directory to write files into (created if missing).
            base_model: Optional base model to restrict downloads to
                (e.g. ``SDXL 1.0``).

        Returns:
            A list of absolute paths to the successfully downloaded files.

        Raises:
            CivitAIDownloadError: If a file's size or SHA256 fails verification.
        """
        data = await self.models_get_async(model_id)
        model = Model(**data)
        downloaded: list[str] = []
        for version in model.model_versions:
            if base_model is not None and version.base_model != base_model:
                continue
            for file in version.files:
                path = await self._download_file_async(file, destination_dir)
                if path:
                    downloaded.append(path)
        return downloaded

    async def _download_file_async(
        self, file: ModelVersionFile, destination_dir: str, *, filename: str | None = None
    ) -> str | None:
        """Download a single model file, resuming and verifying it.

        Downloads to ``<name>.part`` in ``destination_dir``, resuming from the
        existing partial via an HTTP Range request when present. Once the download
        reaches ``int(sizeKB * 1024)`` bytes it is renamed to its final name and,
        if the API provides a SHA256, verified. Returns the final absolute path on
        success, or ``None`` if the download is still incomplete (the ``.part`` is
        kept for a later resume).
        """
        target_name = filename or file.name
        final_path = _os_mod.path.join(destination_dir, target_name)
        part_path = final_path + ".part"
        expected = int(file.size_kb * 1024)
        sha256 = (file.hashes or {}).get("SHA256") if isinstance(file.hashes, dict) else None
        url = file.download_url

        if not url:
            logger.warning("No download URL for file %r; skipping", file.name)
            return None

        _os_mod.makedirs(destination_dir, exist_ok=True)

        if _os_mod.path.exists(final_path):
            if self._file_ok(final_path, expected, sha256):
                return final_path
            raise CivitAIDownloadError(
                f"Existing file {final_path} does not match the expected size/SHA256", final_path
            )

        offset = 0
        hasher = hashlib.sha256()
        if _os_mod.path.exists(part_path):
            offset = _os_mod.path.getsize(part_path)
            if offset >= expected:
                _os_mod.replace(part_path, final_path)
                if self._file_ok(final_path, expected, sha256):
                    return final_path
                raise CivitAIDownloadError(f"Partial download {final_path} failed verification", final_path)
            with open(part_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)

        headers = self.auth_header or {}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True, headers=headers) as session:
            async with session.stream("GET", url) as resp:
                if resp.status_code == 206:
                    write_mode = "ab"
                elif resp.status_code == 200:
                    # Server ignored the Range header; restart from scratch.
                    write_mode = "wb"
                    offset = 0
                    hasher = hashlib.sha256()
                else:
                    raise CivitAIHTTPError(resp.status_code, f"Failed to download {target_name}")

                with open(part_path, write_mode) as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
                        hasher.update(chunk)

        downloaded = _os_mod.path.getsize(part_path)
        if downloaded < expected:
            logger.warning(
                "Incomplete download of %r (%d < %d bytes); left %s for resume",
                target_name,
                downloaded,
                expected,
                part_path,
            )
            return None

        if sha256 and self._sha256_hex(part_path) != sha256:
            raise CivitAIDownloadError(f"SHA256 mismatch for {part_path}", part_path)

        _os_mod.replace(part_path, final_path)
        return final_path

    @staticmethod
    def _sha256_hex(path: str) -> str:
        """Return the uppercase SHA256 hex digest of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest().upper()

    @staticmethod
    def _file_ok(path: str, expected_size: int, sha256: str | None) -> bool:
        """True when a file meets the expected minimum size and, if a hash is
        given, its SHA256 matches."""
        if _os_mod.path.getsize(path) < expected_size:
            return False
        if sha256 and CivitAIClient._sha256_hex(path) != sha256:
            return False
        return True
