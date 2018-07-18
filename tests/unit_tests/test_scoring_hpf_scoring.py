"""Test fucntionalities of hpf scoring."""

from src.scoring.hpf_scoring import HPFScoring
from edward.models import Poisson
import tensorflow as tf


def test_basic_object():
    """Test basic HPF object without datastore."""
    hpf = HPFScoring()
    assert hpf is not None
    assert hpf.theta is None
    assert round(float(hpf.epsilon), 9) == 0.000156564
    assert hpf.theta_dummy is not None
    assert isinstance(hpf.theta_dummy, Poisson)
    assert hpf.theta_dummy.shape.dims[0].value == 13


def test_model_details():
    """Test the basic model details function."""
    hpf = HPFScoring()
    assert hpf is not None
    details = """The model will be scored against
        0 Packages,
        0 Manifests,
        Theta matrix of size 1.52587890625e-05 MB, and
        Beta matrix of size 1.52587890625e-05 MB."""
    assert hpf.model_details() == details


def test_get_sizeof():
    """Test Static _getsizeof method."""
    int_value = 1
    int_size = 2.6702880859375e-05
    assert HPFScoring._getsizeof(int_value) == "{} MB".format(int_size)
