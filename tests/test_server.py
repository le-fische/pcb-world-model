"""Unit tests for the FastAPI web server and game endpoints."""
from fastapi.testclient import TestClient
from silent_trench.server import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "SILENT TRENCH" in response.text


def test_get_maps_endpoint():
    response = client.get("/api/maps")
    assert response.status_code == 200
    maps = response.json()
    assert len(maps) == 10
    names = [m["name"] for m in maps]
    assert "Mariana Trench" in names
    assert "Puerto Rico Trench" in names


def test_game_flow():
    # 1. Start new mission
    new_res = client.post("/api/game/new", json={"map_name": "mariana_trench", "opponent_type": "bot"})
    assert new_res.status_code == 200
    state = new_res.json()
    assert state["status"] == "ONGOING"
    assert state["is_human_turn"] is True
    assert len(state["legal_actions"]) > 0

    # 2. Get state
    st_res = client.get("/api/game/state")
    assert st_res.status_code == 200
    assert st_res.json()["map"]["id"] == "mariana_trench"

    # 3. Submit legal human action (WAIT)
    act_res = client.post("/api/game/action", json={"action_type": "WAIT"})
    assert act_res.status_code == 200
    updated_state = act_res.json()
    assert updated_state["turn"] >= 1
    # Events should be recorded
    assert len(updated_state["events"]) > 0
