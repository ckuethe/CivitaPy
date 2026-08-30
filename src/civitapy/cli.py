"""Command-line tool for downloading model assets from Civitai.

The ``civitapy-dl`` console script downloads one or more models (or single model
versions) into a directory. By default files land under
``<outdir>/<modeltype>/<modelid>_<modelname>_<creatorname>/<basemodel>/`` (the
same layout the :class:`CivitAIClient` uses). With ``--comfyui-models`` they are
placed in a ComfyUI-friendly layout ``<dir>/<comfy_folder>/<modelid>_<modelname>/``
where ``<comfy_folder>`` is the exact ComfyUI subdirectory for the model type
(e.g. ``loras``, ``checkpoints``, ``diffusion_models``); pass ``$COMFYUI_PATH/models``
as the directory.

Inputs may be plain numeric model IDs, ``model:ID`` / ``version:ID`` prefixed
IDs, or Civitai model/version URLs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

from civitapy import __version__
from civitapy.client import CivitAIClient
from civitapy.errors import CivitAIError
from civitapy.models import Model, ModelVersion, ModelVersionFile

__all__ = ["build_plan", "destination_dir", "main", "parse_input"]

# Civitai model/version URL components: ``/models/<id>/<slug>`` and
# ``/model-versions/<id>``; a ``?modelVersionId=<id>`` query targets a version.
_VERSION_PATH_RE = re.compile(r"/model-versions/(\d+)")
_MODEL_PATH_RE = re.compile(r"/models/(\d+)(?:/|$)")
_QUERY_VERSION_RE = re.compile(r"[?&]modelVersionId=(\d+)")


@dataclass
class PlanItem:
    """One file scheduled for download."""

    model: Model
    version: ModelVersion
    file: ModelVersionFile
    dest_dir: str

    @property
    def dest_path(self) -> str:
        return _join_destdir(self.dest_dir, self.file.name)


def parse_input(value: str) -> tuple[str, int]:
    """Parse a CLI input into ``("model" | "version", id)``.

    Accepts a bare numeric model ID, a ``model:ID`` / ``version:ID`` prefix, or
    a Civitai URL whose path is ``/models/<id>`` (optionally with a ``<slug>``
    and a ``?modelVersionId=<id>`` query) or ``/model-versions/<id>``. When a
    URL carries a ``modelVersionId`` query it targets that single version.
    Raises :class:`argparse.ArgumentTypeError` for anything unrecognized.
    """
    stripped = value.strip()
    if match := _QUERY_VERSION_RE.search(stripped):
        return ("version", int(match.group(1)))
    if match := _VERSION_PATH_RE.search(stripped):
        return ("version", int(match.group(1)))
    if match := _MODEL_PATH_RE.search(stripped):
        return ("model", int(match.group(1)))
    if ":" in stripped:
        kind, _, raw_id = stripped.partition(":")
        if kind not in ("model", "version"):
            raise argparse.ArgumentTypeError(f"unknown prefix {kind!r} in {value!r}")
        try:
            return (kind, int(raw_id))
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid id in {value!r}") from None
    try:
        return ("model", int(stripped))
    except ValueError:
        raise argparse.ArgumentTypeError(f"unrecognized input {value!r}") from None


async def build_plan(
    client: CivitAIClient,
    inputs: list[str],
    *,
    destdir: str,
    flat: bool = False,
) -> list[PlanItem]:
    """Resolve every input into a flat list of files to download.

    ``destdir`` roots the output directories. When ``flat`` is ``True`` the
    ComfyUI layout is used (``<destdir>/<type>/<id>_<name>``); otherwise the
    client's default ``<destdir>/<type>/<id>_<name>_<creator>/<basemodel>``
    layout. The client's base-model allow-list is honored for both layouts.

    Each entry in ``inputs`` is either a raw string (parsed by
    :func:`parse_input`) or an already-parsed ``(kind, id)`` tuple — argparse
    ``type=parse_input`` yields the latter.
    """
    items: list[PlanItem] = []
    for value in inputs:
        kind, id_ = value if isinstance(value, tuple) else parse_input(value)
        if kind == "version":
            data = await client.model_versions_get_async(id_)
            version = ModelVersion(**data)
            if not client._should_download_base_model(version.base_model):
                continue
            model_data = await client.models_get_async(version.model_id)
            model = Model(**model_data)
            items.extend(
                PlanItem(
                    model=model,
                    version=version,
                    file=file,
                    dest_dir=destination_dir(client, model, version, destdir=destdir, flat=flat),
                )
                for file in version.files
                if file.download_url
            )
        else:
            data = await client.models_get_async(id_)
            model = Model(**data)
            for version in model.model_versions:
                if not client._should_download_base_model(version.base_model):
                    continue
                items.extend(
                    PlanItem(
                        model=model,
                        version=version,
                        file=file,
                        dest_dir=destination_dir(client, model, version, destdir=destdir, flat=flat),
                    )
                    for file in version.files
                    if file.download_url
                )
    return items


def _join_destdir(destdir: str, *parts: str) -> str:
    """Join a destination prefix with path parts, preserving separator style.

    Literal prefixes like ``/tmp`` keep forward slashes on every platform;
    native Windows paths join natively so files land where the OS expects.
    """
    if "\\" in destdir:
        return os.path.join(destdir, *parts)
    return str(PurePosixPath(destdir, *parts))


def destination_dir(
    client: CivitAIClient,
    model: Model,
    version: ModelVersion,
    *,
    destdir: str,
    flat: bool = False,
) -> str:
    """Compute the destination directory for a version's files.

    ``flat`` produces ``<destdir>/<comfy_folder>/<id>_<name>`` where
    ``comfy_folder`` is the exact, case-sensitive ComfyUI subdirectory for the
    model's type (e.g. ``loras``, ``checkpoints``, ``diffusion_models``). The
    default delegates to the client's canonical version directory.
    """
    if not flat:
        return client._version_download_dir(model, version.base_model, destdir=destdir)
    slug = f"{model.id}_{client._sanitize_component(model.name)}"
    return _join_destdir(destdir, _comfyui_subdir(model.type), slug)


# Mapping from Civitai model types to the exact ComfyUI ``models`` subdirectories.
# Keys are normalized (lowercase, no punctuation); values are case-sensitive and
# must match ComfyUI's folder names.
_COMFYUI_SUBDIR_BY_TYPE = {
    "diffusion": "diffusion_models",
    "unet": "unet",
    "vae": "vae",
    "textencoder": "text_encoders",
    "clip": "clip",
    "checkpoint": "checkpoints",
    "lora": "loras",
    "controlnet": "controlnet",
    "upscale": "upscale_models",
    "textinversion": "embeddings",
    "negativeembedding": "embeddings",
}


def _comfyui_subdir(model_type: str) -> str:
    """Return the exact ComfyUI subdirectory for a Civitai model type.

    Falls back to the sanitized Civitai type when the type is unmapped.
    """
    key = re.sub(r"[^a-z0-9]", "", str(model_type).lower())
    return _COMFYUI_SUBDIR_BY_TYPE.get(key, re.sub(r"[^a-zA-Z0-9._-]", "_", str(model_type)).strip("._-"))


async def _download(item: PlanItem, client: CivitAIClient, *, progress: bool) -> str | None:
    return await client._download_file_async(
        item.file,
        item.dest_dir,
        progress=progress,
        version=item.version,
    )


def _token_from_output_dir(output_dir: str) -> str | None:
    """Read a token from ``<output_dir>/.civitai_token`` if present.

    Returns the trimmed file contents, or ``None`` when the file doesn't exist
    or is empty. An explicit ``--token`` argument always takes precedence.
    """
    path = os.path.join(output_dir, ".civitai_token")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
    except OSError:
        return None
    return token or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="civitapy-dl",
        description="Download model assets from Civitai.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        type=parse_input,
        help="Model/version IDs ('model:123', 'version:456'), bare model IDs, or Civitai URLs.",
    )
    parser.add_argument(
        "-c",
        "--comfyui-models",
        metavar="DIR",
        help="Write files into a ComfyUI models directory layout (<DIR>/<comfy_folder>/<id>_<name>/); pass $COMFYUI_PATH/models as <DIR>.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-file progress bars.",
    )
    parser.add_argument(
        "-n",
        "--no-download",
        action="store_true",
        help="Do not download; print each file and its intended path.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Root directory for downloads (default: current directory).",
    )
    parser.add_argument(
        "-b",
        "--base-model",
        action="append",
        metavar="BASE_MODEL",
        help="Only download versions whose base model matches (repeatable).",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Fail downloads whose SHA256 doesn't match the API hash.",
    )
    parser.add_argument(
        "--token",
        help="Civitai API token (defaults to the CIVITAI_TOKEN environment variable).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version and exit.",
    )
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    destdir = args.comfyui_models or args.output_dir
    flat = args.comfyui_models is not None
    progress = not args.quiet

    token = args.token
    if token is None:
        token = _token_from_output_dir(destdir)

    client = CivitAIClient(
        token=token,
        download_dir=args.output_dir,
        verify_hash=args.verify_hash,
        base_models=args.base_model,
    )

    items = await build_plan(client, args.inputs, destdir=destdir, flat=flat)
    if not items:
        print("Nothing to download.")
        return 0

    if args.no_download:
        for item in items:
            print(f"{item.file.download_url}\t{item.dest_path}")
        return 0

    downloaded = 0
    for item in items:
        path = await _download(item, client, progress=progress)
        if path:
            print(path)
            downloaded += 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    try:
        return asyncio.run(_amain(argv))
    except (CivitAIError, argparse.ArgumentTypeError) as exc:
        print(f"civitapy-dl: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("civitapy-dl: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
