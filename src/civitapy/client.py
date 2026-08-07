from __future__ import annotations

import asyncio
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
        self._base_url = base_url.rstrip("/")
        # Prefer explicit token arg → fall back to CIVITAI_TOKEN env var
        self._token = token or _os_mod.environ.get("CIVITAI_TOKEN")
        self._timeout = httpx.Timeout(timeout)

    @property
    def auth_header(self) -> dict[str, str] | None:
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
        return self._run(self._enums_async())

    async def _enums_async(self) -> dict[str, list[str]]:
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
        return self._run(self.models_get_async(id))

    # -- Model Versions --
    def model_versions_get(self, id: int) -> dict[str, Any]:
        return self._run(self.model_versions_get_async(id))

    def model_versions_by_hash(self, hash_value: str) -> dict[str, Any] | None:
        return self._run(self.model_versions_by_hash_async(hash_value))

    def model_versions_by_hash_bulk(self, hashes: list[str]) -> list[dict[str, Any]]:
        return self._run(self.model_versions_by_hash_bulk_async(hashes))

    def model_versions_by_hash_ids(self, hashes: list[str]) -> list[HashLookupResult]:
        return self._run(self.model_versions_by_hash_ids_async(hashes))

    def model_versions_mini(self, id: int, *, epoch: int | None = None) -> dict[str, Any]:
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
        return self._run(self.collections_list_async(limit=limit, cursor=cursor, query=query, sort=sort, nsfw=nsfw))

    def collections_get(self, id: int) -> dict[str, Any]:
        return self._run(self.collections_get_async(id))

    # -- Creators --
    def creators_list(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        return self._run(self.creators_list_async(limit=limit, page=page, query=query))

    # -- Tags --
    def tags_list(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
        return self._run(self.tags_list_async(limit=limit, page=page, query=query))

    # -- Users --
    def users_me(self) -> dict[str, Any]:
        return self._run(self.users_me_async())

    def users_lookup(
        self,
        *,
        ids: list[int] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
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
        return self._run(
            self.permissions_check_async(entity_ids, entity_type=entity_type, permission=permission, user_id=user_id)
        )

    # -- Vault --
    def vault_get(self) -> dict[str, Any]:
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
        return self._run(self.vault_check_async(model_version_ids))

    def vault_toggle(self, model_version_id: int) -> dict[str, Any]:
        return self._run(self.vault_toggle_async(model_version_id))

    # -----------------------------------------------------------------------
    # Core HTTP (async + sync wrappers delegate to these via _request)
    # -----------------------------------------------------------------------

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, *, json: dict[str, Any] | list | None = None) -> dict[str, Any]:
        return await self._request("POST", path, json=json)

    # -----------------------------------------------------------------------
    # Internal HTTP request with retry logic
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
        return await self._request("GET", path, params=params)

    async def post(self, path: str, *, json: dict[str, Any] | list | None = None) -> dict[str, Any]:
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
        return await self.get(f"/models/{id}")

    # -- Model Versions (async) --
    async def model_versions_get_async(self, id: int) -> dict[str, Any]:
        return await self.get(f"/model-versions/{id}")

    async def model_versions_by_hash_async(self, hash_value: str) -> dict[str, Any] | None:
        try:
            data = await self.get(f"/model-versions/by-hash/{hash_value}")
            return data if isinstance(data, dict) else None
        except CivitAINotFoundError:
            return None

    async def model_versions_by_hash_bulk_async(self, hashes: list[str]) -> list[dict[str, Any]]:
        if len(hashes) > 100:
            raise ValueError("Maximum 100 hashes per request")
        data = await self.post("/model-versions/by-hash", json=hashes)
        return data if isinstance(data, list) else []

    async def model_versions_by_hash_ids_async(self, hashes: list[str]) -> list[HashLookupResult]:
        if len(hashes) > 10_000:
            raise ValueError("Maximum 10,000 hashes per request")
        data = await self.post("/model-versions/by-hash/ids", json=hashes)
        return [HashLookupResult(**item) for item in (data if isinstance(data, list) else [])]

    async def model_versions_mini_async(self, id: int, *, epoch: int | None = None) -> dict[str, Any]:
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
        params = _clean_params(
            {"limit": limit or 100, "cursor": cursor, "query": query, "sort": sort, "nsfw": nsfw or None}
        )
        return await self.get("/collections", params=params)

    async def collections_get_async(self, id: int) -> dict[str, Any]:
        return await self.get(f"/collections/{id}")

    # -- Creators (async) --
    async def creators_list_async(
        self,
        *,
        limit: int = 20,
        page: int = 1,
        query: str | None = None,
    ) -> dict[str, Any]:
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
        params = _clean_params({"limit": limit or 20, "page": page, "query": query})

        if page * (params.get("limit") or 20) > 1000:
            raise CivitAIRateLimitError("You've requested too many pages, please use cursors instead")

        return await self.get("/tags", params=params)

    # -- Users (async) --
    async def users_me_async(self) -> dict[str, Any]:
        return await self.get("/me")

    async def users_lookup_async(
        self,
        *,
        ids: list[int] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
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
        params = _clean_params({"modelVersionIds": ",".join(str(i) for i in model_version_ids)})
        data = await self.get("/vault/check-vault", params=params)
        return data if isinstance(data, list) else []

    async def vault_toggle_async(self, model_version_id: int) -> dict[str, Any]:
        return await self.post("/vault/toggle-version", params={"modelVersionId": model_version_id})
