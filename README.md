# CivitaPy — Python bindings for the CivitAI Site API

This package provides a python interface to [civitai](https://developer.civitai.com/site/);
a popular sharing platform for image generation models...
[thenoise](https://github.com/lemonade-sdk/thenoise)
made me do it.

Brought to you by [opencode](https://github.com/anomalyco/opencode/),
[lemonade](https://github.com/lemonade-sdk/lemonade/),
[Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF),
[Deepseek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF),
and a bunch of careful prompts.

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
mv = ModelVersion(**client.model_versions_get(1331249))
print(mv.air)  # 'urn:air:sdxl:lora:civitai:1182863@1331249'

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
    mv = await client.model_versions_get_async(1331249)
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
`int(sizeKB * 1024)` and, once size is acceptable, its SHA256.

```python
client = CivitAIClient()
paths = client.download_model_version(1331249)
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
This is useful when a machine can only run "Anima" models (for example), so
downloading assets for any other base model would be pointless.
Matching ignores case, punctuation and whitespace, so Civitai's `ZImageTurbo`
matches a config value `Z-Image-Turbo`:

```python
client = CivitAIClient(
    base_models=["Flux.2 Klein 4B", "Z-Image-Turbo", "Anima"],
)
```

With a filter set, `download_model` / `download_model_version` only fetch files
whose version's base model matches one of the listed entries.

### CLI tool

A `civitapy-dl` console script is installed alongside the package for
downloading model assets from the shell. Inputs can be bare model IDs,
`model:ID` / `version:ID` prefixes, or Civitai model/version URLs:

```bash
# Download model 827184 (all versions) into the current directory
civitapy-dl 827184

# Download one specific version, into a ComfyUI models directory layout
civitapy-dl version:1331249 -c ~/ComfyUI/models

# Just list the files that would be downloaded and their target paths
civitapy-dl https://civitai.com/models/827184 -n
```

Without `-c`, files land under
`<outdir>/<modeltype>/<modelid>_<modelname>_<creatorname>/<basemodel>/`. With
`-c DIR`, files are placed in a ComfyUI-friendly layout
`<DIR>/<modeltype>/<modelid>_<modelname>/<file>`, suitable for pointing ComfyUI
at. `-q` suppresses per-file progress bars, `-n` performs a dry run (printing
each source URL and its destination path without downloading), `-b` restricts
downloads to matching base models (repeatable), and `--verify-hash` opts into
strict SHA256 verification.

### Gated / early-access files

Per the Civitai API docs, file downloads require a bearer token (401 without
one) and early-access or private resources return 403 for callers without an
active membership. These raise detailed `CivitAIAuthError` / `CivitAIForbiddenError`
messages instead of a generic HTTP error.
