import hashlib

from civitapy import CivitAIClient


def _write(path, data):
    path.write_bytes(data)
    return str(path)


def _sha256_hex_bytes(data):
    return hashlib.sha256(data).hexdigest().upper()


def test_file_problem_ok(tmp_path):
    data = b"a" * 100
    p = _write(tmp_path / "f", data)
    assert CivitAIClient._file_problem(p, 100, _sha256_hex_bytes(data)) is None


def test_file_problem_size_mismatch(tmp_path):
    p = _write(tmp_path / "f", b"a" * 50)
    problem = CivitAIClient._file_problem(p, 100, None)
    assert problem is not None
    assert "size mismatch" in problem
    assert problem.endswith(")")  # message is well-formed


def test_file_problem_hash_mismatch(tmp_path):
    p = _write(tmp_path / "f", b"a" * 100)
    problem = CivitAIClient._file_problem(p, 100, "A" * 64)
    assert problem == "SHA256 hash mismatch"


def test_file_problem_ignores_hash_when_too_small(tmp_path):
    p = _write(tmp_path / "f", b"a" * 10)
    problem = CivitAIClient._file_problem(p, 100, "A" * 64)
    assert "size mismatch" in problem  # size shortfall reported first


def test_file_ok_true_when_matches(tmp_path):
    data = b"b" * 200
    p = _write(tmp_path / "f", data)
    assert CivitAIClient._file_ok(p, 200, _sha256_hex_bytes(data)) is True


def test_file_ok_false_on_hash_mismatch(tmp_path):
    p = _write(tmp_path / "f", b"b" * 200)
    assert CivitAIClient._file_ok(p, 200, "F" * 64) is False


def test_file_ok_false_on_size_mismatch(tmp_path):
    p = _write(tmp_path / "f", b"b" * 10)
    assert CivitAIClient._file_ok(p, 200, None) is False


def test_sha256_hex(tmp_path):
    data = b"hello"
    p = _write(tmp_path / "f", data)
    assert CivitAIClient._sha256_hex(p) == hashlib.sha256(data).hexdigest().upper()
