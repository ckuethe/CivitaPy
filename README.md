# CivitaPy — Python bindings for the CivitAI Site API

A play on **Civitai**, **API**, and **py**.

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
from civitapy import CivitAIClient, Model

client = CivitAIClient()
for model in client.models_list_paginated(limit=100):  # async generator
    print(model.name)
```
