import argparse
import asyncio

import pytest
from conftest import FakeResponse, mock_async_client

import civitapy
from civitapy import CivitAIClient
from civitapy.cli import _token_from_output_dir, build_parser, build_plan, destination_dir, parse_input


def run(coro):
    return asyncio.run(coro)


def _model_json(model_id=42, name="my model", type="Checkpoint", base_model="SDXL 1.0"):
    return {
        "id": model_id,
        "name": name,
        "type": type,
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
                        "sizeKB": 1.0,
                        "downloadUrl": "https://example.com/dl",
                    }
                ],
            }
        ],
    }


def _version_json(model_id=42, version_id=99, base_model="SDXL 1.0"):
    return {
        "id": version_id,
        "modelId": model_id,
        "name": "v1",
        "baseModel": base_model,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "files": [
            {
                "id": 1,
                "name": "m.bin",
                "type": "Model",
                "sizeKB": 1.0,
                "downloadUrl": "https://example.com/dl",
            }
        ],
    }


def test_parse_input_bare_id():
    assert parse_input("42") == ("model", 42)


def test_parse_input_prefixes():
    assert parse_input("model:42") == ("model", 42)
    assert parse_input("version:99") == ("version", 99)


def test_parse_input_urls():
    assert parse_input("https://civitai.com/models/42") == ("model", 42)
    assert parse_input("https://civitai.com/model-versions/99") == ("version", 99)


def test_parse_input_slug_urls():
    # Slug URLs without a version query target the model.
    assert parse_input("https://civitai.com/models/1331249/bubbli-cartoon-il") == ("model", 1331249)
    # With a modelVersionId query they target that single version.
    assert parse_input("1331249/bubbli-cartoon-il?modelVersionId=1503014") == ("version", 1503014)
    assert parse_input("https://civitai.com/models/1331249/bubbli-cartoon-il?modelVersionId=1503014") == (
        "version",
        1503014,
    )
    # A modelVersionId query always selects that single version.
    assert parse_input("https://civitai.com/model-versions/99?modelVersionId=1503014") == ("version", 1503014)


def test_parse_input_invalid():
    for bad in ["", "foo", "model:abc", "unknown:1", "not a url"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_input(bad)


def test_destination_dir_flat(client, tmp_path):
    from civitapy.models import Model, ModelVersion

    model = Model(**{**_model_json(), "modelVersions": []})
    version = ModelVersion(**{**_version_json(), "files": []})

    dest = destination_dir(client, model, version, destdir=str(tmp_path), flat=True)
    assert dest == str(tmp_path / "checkpoints" / "42_my_model")

    dest = destination_dir(client, model, version, destdir=str(tmp_path), flat=False)
    assert dest == str(tmp_path / "Checkpoint" / "42_my_model_some_creator" / "SDXL_1.0")


@pytest.mark.parametrize(
    "type_,folder",
    [
        ("Checkpoint", "checkpoints"),
        ("LORA", "loras"),
        ("VAE", "vae"),
        ("CLIP", "clip"),
        ("UNet", "unet"),
        ("Diffusion", "diffusion_models"),
        ("Text Encoder", "text_encoders"),
        ("ControlNet", "controlnet"),
        ("Upscale", "upscale_models"),
        ("Text Inversion", "embeddings"),
        ("Negative Embedding", "embeddings"),
    ],
)
def test_destination_dir_flat_maps_comfyui_folders(client, tmp_path, type_, folder):
    from civitapy.models import Model, ModelVersion

    model = Model(**{**_model_json(type=type_), "modelVersions": []})
    version = ModelVersion(**{**_version_json(), "files": []})

    dest = destination_dir(client, model, version, destdir=str(tmp_path), flat=True)
    assert dest == str(tmp_path / folder / "42_my_model")


def test_destination_dir_flat_unmapped_type_falls_back(client, tmp_path):
    from civitapy.models import Model, ModelVersion

    model = Model(**{**_model_json(type="Some Odd Type"), "modelVersions": []})
    version = ModelVersion(**{**_version_json(), "files": []})

    dest = destination_dir(client, model, version, destdir=str(tmp_path), flat=True)
    assert dest == str(tmp_path / "Some_Odd_Type" / "42_my_model")


def test_build_plan_model_input(client, tmp_path):
    model_resp = FakeResponse(json_data=_model_json(), content=b"{}")
    with mock_async_client(model_resp):
        items = run(build_plan(client, ["42"], destdir=str(tmp_path), flat=True))

    assert len(items) == 1
    assert items[0].file.name == "m.bin"
    assert items[0].dest_path == str(tmp_path / "checkpoints" / "42_my_model" / "m.bin")


def test_build_plan_version_input(client, tmp_path):
    version_resp = FakeResponse(json_data=_version_json(), content=b"{}")
    model_resp = FakeResponse(json_data=_model_json(), content=b"{}")
    with mock_async_client([version_resp, model_resp]):
        items = run(build_plan(client, ["version:99"], destdir=str(tmp_path), flat=True))

    assert len(items) == 1
    assert items[0].dest_path == str(tmp_path / "checkpoints" / "42_my_model" / "m.bin")


def test_build_plan_accepts_parsed_tuples(client, tmp_path):
    # argparse `type=parse_input` hands tuples to build_plan directly.
    version_resp = FakeResponse(json_data=_version_json(), content=b"{}")
    model_resp = FakeResponse(json_data=_model_json(), content=b"{}")
    with mock_async_client([version_resp, model_resp]):
        items = run(build_plan(client, [("version", 99)], destdir=str(tmp_path), flat=True))

    assert len(items) == 1
    assert items[0].dest_path == str(tmp_path / "checkpoints" / "42_my_model" / "m.bin")


def test_build_plan_respects_base_model_filter(tmp_path):
    client = CivitAIClient(base_url="https://example.com", download_dir=str(tmp_path), base_models=["SDXL 1.0"])
    model_resp = FakeResponse(json_data=_model_json(base_model="SDXL 1.0"), content=b"{}")
    with mock_async_client(model_resp):
        items = run(build_plan(client, ["42"], destdir=str(tmp_path), flat=True))
    assert len(items) == 1

    client = CivitAIClient(base_url="https://example.com", download_dir=str(tmp_path), base_models=["Flux.2 Klein 4B"])
    model_resp = FakeResponse(json_data=_model_json(base_model="SDXL 1.0"), content=b"{}")
    with mock_async_client(model_resp):
        items = run(build_plan(client, ["42"], destdir=str(tmp_path), flat=True))
    assert items == []


def test_token_from_output_dir_missing(tmp_path):
    assert _token_from_output_dir(str(tmp_path)) is None


def test_token_from_output_dir_reads_file(tmp_path):
    (tmp_path / ".civitai_token").write_text("  abc123\n")
    assert _token_from_output_dir(str(tmp_path)) == "abc123"


def test_token_from_output_dir_empty(tmp_path):
    (tmp_path / ".civitai_token").write_text("   \n")
    assert _token_from_output_dir(str(tmp_path)) is None


def test_package_version_is_pep440_string():
    import importlib.metadata

    assert civitapy.__version__ == importlib.metadata.version("civitapy")
    assert civitapy.__version__.count(".") >= 1


def test_cli_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert civitapy.__version__ in capsys.readouterr().out
