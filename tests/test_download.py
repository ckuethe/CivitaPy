import asyncio
import hashlib

import pytest
from conftest import FakeResponse, mock_async_client

from civitapy import CivitAIClient, Model, ModelVersion, ModelVersionFile
from civitapy.errors import (
    CivitAIAuthError,
    CivitAIDownloadError,
    CivitAIForbiddenError,
    CivitAINotFoundError,
)


def run(coro):
    return asyncio.run(coro)


def _file(name="m.bin", size_kb=1.0, sha=None):
    return ModelVersionFile(
        id=1,
        name=name,
        type="Model",
        sizeKB=size_kb,
        hashes={"SHA256": sha} if sha else None,
        downloadUrl="https://example.com/dl",
    )


def _sha(data):
    return hashlib.sha256(data).hexdigest().upper()


def _attempt(client, file, final_path, part_path, expected, sha, url, content, status=200, **kwargs):
    resp = FakeResponse(status, content=content)
    with mock_async_client(resp):
        return run(
            client._download_file_attempt(
                file, final_path, part_path, expected, sha, url, verify_hash=kwargs.pop("verify_hash", None)
            )
        )


def test_download_fresh_200(client, tmp_path):
    data = b"x" * 1024
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file(size_kb=1.0, sha=_sha(data))
    result = _attempt(client, file, final, part, 1024, _sha(data), "https://example.com/dl", data)
    assert result == final
    assert (tmp_path / "m.bin").read_bytes() == data
    assert not (tmp_path / "m.bin.part").exists()


def test_download_resumes_partial(client, tmp_path):
    data = b"y" * 1024
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    (tmp_path / "m.bin.part").write_bytes(data[:512])

    file = _file(size_kb=1.0, sha=_sha(data))
    resp = FakeResponse(206, content=data[512:])  # server honors Range
    with mock_async_client(resp):
        result = run(
            client._download_file_attempt(
                file, final, part, 1024, _sha(data), "https://example.com/dl", verify_hash=True
            )
        )
    assert result == final
    assert (tmp_path / "m.bin").read_bytes() == data


def test_download_incomplete_returns_none(client, tmp_path):
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file(size_kb=1.0)
    result = _attempt(client, file, final, part, 1024, None, "https://example.com/dl", b"tiny")
    assert result is None
    assert (tmp_path / "m.bin.part").exists()  # kept for resume


def test_download_hash_mismatch_raises_when_verify(client, tmp_path):
    data = b"z" * 1024
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file(size_kb=1.0)
    with pytest.raises(CivitAIDownloadError) as exc:
        _attempt(client, file, final, part, 1024, _sha(b"different"), "https://example.com/dl", data, verify_hash=True)
    assert "SHA256" in str(exc.value)
    assert not (tmp_path / "m.bin.part").exists()  # bad part removed


def test_download_hash_mismatch_accepted_when_not_verify(client, tmp_path):
    data = b"w" * 1024
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file(size_kb=1.0)
    result = _attempt(
        client, file, final, part, 1024, _sha(b"different"), "https://example.com/dl", data, verify_hash=False
    )
    assert result == final


def test_download_404_raises(client, tmp_path):
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file()
    with pytest.raises(CivitAINotFoundError):
        _attempt(client, file, final, part, 1024, None, "https://example.com/dl", b"", status=404)


def test_download_401_raises_auth_error(client, tmp_path):
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file()
    with pytest.raises(CivitAIAuthError):
        _attempt(client, file, final, part, 1024, None, "https://example.com/dl", b"", status=401)


def test_download_403_raises_forbidden(client, tmp_path):
    final = str(tmp_path / "m.bin")
    part = final + ".part"
    file = _file()
    with pytest.raises(CivitAIForbiddenError):
        _attempt(client, file, final, part, 1024, None, "https://example.com/dl", b"", status=403)


# -- destdir override --


def _make_model(tmp_path, base_model="SDXL 1.0", size_kb=1.0, name="my model"):
    data = {
        "id": 42,
        "name": name,
        "type": "Checkpoint",
        "creator": {"id": 7, "username": "some creator"},
        "modelVersions": [
            {
                "id": 99,
                "name": "v1",
                "baseModel": base_model,
                "publishedAt": "2024-01-01T00:00:00Z",
                "files": [
                    {
                        "id": 1,
                        "name": "m.bin",
                        "type": "Model",
                        "sizeKB": size_kb,
                        "downloadUrl": "https://example.com/dl",
                    }
                ],
            }
        ],
    }
    return Model(**data)


def test_model_download_dir_override(client, tmp_path):
    model = _make_model(tmp_path)
    default_dir = client._model_download_dir(model)
    assert default_dir == str(tmp_path / "Checkpoint" / "42_my_model_some_creator")

    override_dir = client._model_download_dir(model, destdir="/tmp")
    assert override_dir == "/tmp/Checkpoint/42_my_model_some_creator"
    assert override_dir != default_dir


def test_model_version_summary_missing_published_at():
    data = {
        "id": 42,
        "name": "x",
        "type": "Checkpoint",
        "modelVersions": [{"id": 20, "name": "v", "baseModel": "SDXL 1.0", "index": 2}],
    }
    model = Model(**data)
    assert model.model_versions[0].published_at is None


def test_model_version_missing_published_at():
    data = {
        "id": 20,
        "modelId": 42,
        "name": "v1",
        "baseModel": "SDXL 1.0",
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "files": [],
    }
    version = ModelVersion(**data)
    assert version.published_at is None


def test_version_download_dir_override(client, tmp_path):
    model = _make_model(tmp_path)
    default_dir = client._version_download_dir(model, "SDXL 1.0")
    assert default_dir == str(tmp_path / "Checkpoint" / "42_my_model_some_creator" / "SDXL_1.0")

    override_dir = client._version_download_dir(model, "SDXL 1.0", destdir="/tmp")
    assert override_dir == "/tmp/Checkpoint/42_my_model_some_creator/SDXL_1.0"


def test_download_model_async_destdir(client, tmp_path):
    model = _make_model(tmp_path, size_kb=1.0)
    data = b"d" * 1024
    model_resp = FakeResponse(json_data=model.model_dump(mode="json", by_alias=True), content=b"{}")
    file_resp = FakeResponse(200, content=data)
    with mock_async_client([model_resp, file_resp]):
        paths = run(client.download_model_async(42, destdir="/tmp"))

    assert len(paths) == 1
    assert paths[0].startswith("/tmp/")
    assert paths[0] == "/tmp/Checkpoint/42_my_model_some_creator/SDXL_1.0/m.bin"
    with open(paths[0], "rb") as f:
        assert f.read() == data


def test_download_version_downloads_all_related_files_next_to_each_other(client, tmp_path):
    """A version with several files (e.g. a .safetensors plus a workflow .json)
    downloads every file into the same directory, next to the primary file."""
    version_data = {
        "id": 99,
        "modelId": 42,
        "name": "v1",
        "baseModel": "SDXL 1.0",
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "files": [
            {
                "id": 1,
                "name": "model.safetensors",
                "type": "Model",
                "sizeKB": 1.0,
                "primary": True,
                "downloadUrl": "https://example.com/dl?type=Model",
            },
            {
                "id": 2,
                "name": "workflow.json",
                "type": "Config",
                "sizeKB": 1.0,
                "downloadUrl": "https://example.com/dl?type=Config",
            },
        ],
    }
    model_data = {
        "id": 42,
        "name": "my model",
        "type": "LORA",
        "creator": {"id": 7, "username": "some creator"},
        "modelVersions": [],
    }
    safetensors = b"s" * 1024
    workflow = b"w" * 1024
    version_resp = FakeResponse(json_data=version_data, content=b"{}")
    model_resp = FakeResponse(json_data=model_data, content=b"{}")
    file_resp = FakeResponse(200, content=safetensors)
    config_resp = FakeResponse(200, content=workflow)
    with mock_async_client([version_resp, model_resp, file_resp, config_resp]):
        paths = run(client.download_model_version_async(99))

    assert len(paths) == 2
    dest = str(tmp_path / "LORA" / "42_my_model_some_creator" / "SDXL_1.0")
    assert paths[0] == dest + "/model.safetensors"
    assert paths[1] == dest + "/workflow.json"
    # Both files land side by side in the version directory.
    assert (tmp_path / "LORA" / "42_my_model_some_creator" / "SDXL_1.0").is_dir()
    assert (tmp_path / "LORA" / "42_my_model_some_creator" / "SDXL_1.0" / "model.safetensors").read_bytes() == safetensors
    assert (tmp_path / "LORA" / "42_my_model_some_creator" / "SDXL_1.0" / "workflow.json").read_bytes() == workflow


def _filtered_client(tmp_path):
    return CivitAIClient(
        base_url="https://example.com",
        download_dir=str(tmp_path),
        retry_count=3,
        min_request_interval=0.0,
        base_models=["SDXL 1.0"],
    )


def test_download_model_async_star_bypasses_base_model_filter(tmp_path):
    """base_model='*' downloads every version even when its base model is
    excluded by the client's global allow-list (e.g. small workflow files)."""
    client = _filtered_client(tmp_path)
    model = _make_model(tmp_path, base_model="LTXV 2.3", name="workflow pack")
    data = b"d" * 1024
    model_resp = FakeResponse(json_data=model.model_dump(mode="json", by_alias=True), content=b"{}")
    file_resp = FakeResponse(200, content=data)
    with mock_async_client([model_resp, file_resp]):
        paths = run(client.download_model_async(42, base_model="*"))

    assert len(paths) == 1
    assert paths[0] == str(tmp_path / "Checkpoint" / "42_workflow_pack_some_creator" / "LTXV_2.3" / "m.bin")


def test_download_model_async_star_respects_concrete_override(tmp_path):
    """A concrete base_model still filters even with the '*' sentinel available."""
    client = _filtered_client(tmp_path)
    model = _make_model(tmp_path, base_model="LTXV 2.3", name="workflow pack")
    model_resp = FakeResponse(json_data=model.model_dump(mode="json", by_alias=True), content=b"{}")
    with mock_async_client([model_resp]):
        paths = run(client.download_model_async(42, base_model="SDXL 1.0"))

    assert paths == []


def test_download_version_async_star_bypasses_base_model_filter(tmp_path):
    """base_model='*' lets a single version download despite the allow-list."""
    client = _filtered_client(tmp_path)
    version_data = {
        "id": 99,
        "modelId": 42,
        "name": "v1",
        "baseModel": "LTXV 2.3",
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "files": [
            {
                "id": 1,
                "name": "workflow.json",
                "type": "Config",
                "sizeKB": 1.0,
                "downloadUrl": "https://example.com/dl?type=Config",
            }
        ],
    }
    model_data = {
        "id": 42,
        "name": "workflow pack",
        "type": "ComfyWorkflows",
        "creator": {"id": 7, "username": "some creator"},
        "modelVersions": [],
    }
    data = b"w" * 1024
    version_resp = FakeResponse(json_data=version_data, content=b"{}")
    model_resp = FakeResponse(json_data=model_data, content=b"{}")
    file_resp = FakeResponse(200, content=data)
    with mock_async_client([version_resp, model_resp, file_resp]):
        paths = run(client.download_model_version_async(99, base_model="*"))

    assert len(paths) == 1
    assert paths[0] == str(tmp_path / "ComfyWorkflows" / "42_workflow_pack_some_creator" / "LTXV_2.3" / "workflow.json")
