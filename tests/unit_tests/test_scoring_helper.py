"""Test static and class methods of HPFScoring."""
from src.scoring.hpf_scoring import HPFScoring


def test_get_sizeof():
    """Test Static _getsizeof method."""
    int_value = 1
    int_size = 2.6702880859375e-05
    assert HPFScoring._getsizeof(int_value) == "{} MB".format(int_size)
