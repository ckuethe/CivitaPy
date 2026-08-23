import asyncio

from conftest import FakeResponse, mock_async_client

from civitapy import AsyncPaginator, Model


def run(coro):
    return asyncio.run(coro)


def _page(items, next_cursor=None, total_pages=None, next_page=None):
    metadata = {}
    if next_cursor is not None:
        metadata["nextCursor"] = next_cursor
    if total_pages is not None:
        metadata["totalPages"] = total_pages
    if next_page is not None:
        metadata["nextPage"] = next_page
    return FakeResponse(200, {"items": items, "metadata": metadata})


def _model_item(i):
    return {
        "id": i,
        "name": f"m{i}",
        "type": "Checkpoint",
        "nsfw": False,
        "creator": {"username": "u", "image": None},
        "modelVersions": [],
        "tags": [],
    }


def test_cursor_paginator_iterates_pages(client):
    p1 = _page([_model_item(1)], next_cursor="c2")
    p2 = _page([_model_item(2)])  # no nextCursor -> stop
    with mock_async_client([p1, p2]):
        pag = AsyncPaginator(client, "/models", Model, limit=100)
        models = run(_collect(pag))
    assert [m.id for m in models] == [1, 2]


def test_cursor_paginator_sends_cursor(client):
    seen = []

    async def record(method, url, **kwargs):
        seen.append(kwargs.get("params"))
        idx = len(seen) - 1
        if idx == 0:
            return _page([_model_item(1)], next_cursor="c2")
        return _page([_model_item(2)])

    from unittest import mock

    fake_session = mock.MagicMock()
    fake_session.request = record
    fake_client = mock.MagicMock()
    fake_client.__aenter__.return_value = fake_session
    fake_client.__aexit__.return_value = False

    with mock.patch("civitapy.client.httpx.AsyncClient", return_value=fake_client):
        run(_collect(AsyncPaginator(client, "/models", Model, limit=100)))

    # First call has no cursor; second call carries it.
    assert "cursor" not in (seen[0] or {})
    assert seen[1].get("cursor") == "c2"


def test_page_paginator_iterates_pages(client):
    p1 = _page([_model_item(1)], next_page=True, total_pages=2)
    p2 = _page([_model_item(2)], next_page=True, total_pages=2)
    with mock_async_client([p1, p2]):
        pag = AsyncPaginator(client, "/models", Model, limit=100, page=1)
        models = run(_collect(pag))
    assert [m.id for m in models] == [1, 2]


def test_page_paginator_stops_at_total_pages(client):
    p1 = _page([_model_item(1)], next_page=True, total_pages=1)
    with mock_async_client([p1]):
        pag = AsyncPaginator(client, "/models", Model, limit=100, page=1)
        models = run(_collect(pag))
    assert [m.id for m in models] == [1]


def test_models_list_paginated_async_returns_paginator(client):
    p = client.models_list_paginated_async(limit=100)
    assert isinstance(p, AsyncPaginator)


async def _collect(paginator):
    return [item async for item in paginator]
