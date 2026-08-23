# CivitaPy — Python bindings for the CivitAI Site API

Brought to you by opencode, lemonade, and Qwen3.6-35B-A3B, and a couple
of careful prompts.

## Install

```bash
pip install civitapy
```

## Quick start

```python
from civitapy import CivitAIClient, ModelVersion

client = CivitAIClient()  # reads CIVITAI_TOKEN from env automatically

# Public endpoints (no token needed)
models = client.models_list(limit=5)
mv = ModelVersion(**client.model_versions_get(2514310))
print(mv.air)  # urn:air:sdxl:checkpoint:civitai:827184@2514310

# Authenticated endpoints (token from env or explicit arg)
user = client.users_me()  # requires CIVITAI_TOKEN
```

## Async API

Every sync method has an `_async` counterpart for use in async contexts:

```python
import asyncio
from civitapy import CivitAIClient, ModelVersion

async def main():
    client = CivitAIClient(token="my-token")
    mv = await client.model_versions_get_async(2514310)
    print(mv["name"])

asyncio.run(main())
```

## Pagination

Cursor-based paginators for all list endpoints:

```python
from civitapy import CivitAIClient

client = CivitAIClient()
async for model in client.models_list_paginated_async(limit=100):
    print(model.name)
```

## Downloads

Downloads resume interrupted transfers, retry transient failures (429 / 5xx /
network errors) with exponential backoff, verify each file's size against
`int(sizeKB * 1024)` and, once size is acceptable, its SHA256:

```python
client = CivitAIClient()
paths = client.download_model_version(2514310)
print(paths)  # absolute paths to the downloaded files
```

Retries are configurable in the initializer (default 3). A best-effort internal
rate limiter also honors `X-RateLimit-*` / `Retry-After` headers and can space
requests with a minimum interval:

```python
client = CivitAIClient(retry_count=5, min_request_interval=0.5)
```

### Base-model filter

Configure a global allow-list of base models so downloads skip anything else.
Matching ignores case, punctuation and whitespace, so Civitai's `ZImageTurbo`
matches a config value `Z-Image-Turbo`:

```python
client = CivitAIClient(
    base_models=["Flux.2 Klein 4B", "Z-Image-Turbo", "Anima"],
)
```

With a filter set, `download_model` / `download_model_version` only fetch files
whose version's base model matches one of the listed entries.

### Gated / early-access files

Per the Civitai API docs, file downloads require a bearer token (401 without
one) and early-access or private resources return 403 for callers without an
active membership. These raise detailed `CivitAIAuthError` / `CivitAIForbiddenError`
messages instead of a generic HTTP error.
