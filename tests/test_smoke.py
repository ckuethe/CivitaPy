import sys

import pytest

sys.path.insert(0, "../src")  # Add src to path when running from tests/ dir

from civitapy import CivitAIClient, ModelVersion, MiniModelVersion, TagItem


@pytest.mark.network
def test_public_endpoints():
    client = CivitAIClient()

    data = client.models_list(limit=2)
    items = data.get("items", [])
    print(f"models_list returned {len(items)} models")

    data = client.tags_list(limit=5, page=1)
    tag_items = [TagItem(**i) for i in data.get("items", [])]
    assert len(tag_items) == 5
    print(f"tags_list returned {len(tag_items)} tags: {[t.name for t in tag_items]}")

    mv_data = client.model_versions_get(2514310)
    mv = ModelVersion(**mv_data)
    assert mv.air is not None
    print(f"model_versions_get: id={mv.id}, name='{mv.name}', AIR={mv.air}")

    mini = client.model_versions_mini(2514310)
    mv_mini = MiniModelVersion(**mini)
    assert mv_mini.can_generate is True or mv_mini.can_generate is False  # just verify it parses
    print(f"model_versions_mini: name='{mv_mini.version_name}', format={mv_mini.format}")


@pytest.mark.network
def test_auth_endpoint():
    client = CivitAIClient()
    user_data = client.users_me()
    assert "id" in user_data and "username" in user_data
    print(f"users_me: id={user_data['id']}, username='{user_data['username']}'")


if __name__ == "__main__":
    test_public_endpoints()
    test_auth_endpoint()
