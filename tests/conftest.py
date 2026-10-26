import re
import os

import numpy as np
import pytest

from maintainai.database import connect, initialize
from maintainai.service import MaintenanceStore


@pytest.fixture(autouse=True)
def isolate_local_llm_settings(tmp_path, monkeypatch):
    """Never read a developer's real project-root .env, even in cloud fallback tests."""
    monkeypatch.setattr("maintainai.llm.PROJECT_ROOT", tmp_path)


@pytest.fixture(autouse=True)
def block_cloud_network(monkeypatch):
    """Never use a developer's real credentials or make unmocked HTTP calls in tests."""
    for name in list(os.environ):
        if name.startswith("MAINTAINAI_LLM_"):
            monkeypatch.delenv(name)

    def blocked(*args, **kwargs):
        raise AssertionError("Real HTTP calls are blocked in routine tests; inject a fake provider or transport.")

    monkeypatch.setattr("requests.sessions.Session.request", blocked)


@pytest.fixture(autouse=True)
def block_real_model_loads(monkeypatch):
    """Routine tests must never download weights or start real model inference."""
    from maintainai import embeddings

    original = embeddings._load_model

    def blocked(*args, **kwargs):
        raise AssertionError("Use fake embeddings in routine tests; real model loading is blocked.")

    monkeypatch.setattr(embeddings, "_load_model", blocked)
    return original


class FakeEmbedder:
    model_key = "deterministic-test-model:field-mean-v1"
    dimension = 6
    vocabulary = (
        {"vacuum", "suction", "gripper", "cup", "seal", "pressure", "leak", "drops"},
        {"robot", "robotic", "wrist", "joint", "torque", "encoder", "vibration"},
        {"zaber", "gantry", "homing", "carriage", "travel", "axis", "steps"},
        {"conveyor", "belt", "roller", "photoeye", "accumulation", "pulley"},
        {"vision", "camera", "image", "focus", "lens", "illumination", "brightness"},
    )

    def __init__(self):
        self.calls = []
        self.hook = None

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        vectors = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            vectors.append([len(words & group) for group in self.vocabulary] + [0.01])
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def store(db_path):
    with connect(db_path) as connection:
        initialize(connection)
        yield MaintenanceStore(connection)


@pytest.fixture
def equipment_id(store):
    return store.create_equipment(asset_tag="TEST-001", name="Test robot", category="Robotics")
