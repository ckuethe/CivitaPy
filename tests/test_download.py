import asyncio
import hashlib

import pytest

from civitapy.errors import (
    CivitAIAuthError,
    CivitAIDownloadError,
    CivitAIForbiddenError,
    CivitAINotFoundError,
)

from civitapy import ModelVersionFile

from conftest import FakeResponse, mock_async_client


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
