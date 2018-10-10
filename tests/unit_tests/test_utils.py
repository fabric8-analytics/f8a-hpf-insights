"""Test helper functions."""

from src.utils import convert_string2bool_env


def test_convert_string2bool_env():
    """Test convert_string2bool_env()."""
    assert convert_string2bool_env("TRUE")
    assert not convert_string2bool_env("False")
