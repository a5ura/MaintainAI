import pytest

from maintainai.database import connect, initialize
from maintainai.service import MaintenanceStore


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
