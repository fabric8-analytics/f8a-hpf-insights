"""Tests the LocalDataStore."""

from src.data_store.local_data_store import LocalDataStore


def test_local_datastore():
    """Tests local datastore object and name."""
    local_obj = LocalDataStore("/tmp")
    assert local_obj is not None
    assert local_obj.get_name() == "Local filesytem dir: /tmp"
