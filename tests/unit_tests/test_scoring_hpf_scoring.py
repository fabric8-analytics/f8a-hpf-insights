"""Test functionalities of hpf scoring."""
import unittest

from src.flask_endpoint import app
from src.data_store.local_data_store import LocalDataStore
from src.scoring.hpf_scoring import HPFScoring


class TestHPFScoringMethods(unittest.TestCase):
    """Test functionalities of hpf scoring."""

    def __init__(self, *args, **kwargs):
        """Initialise the local data store and HPF object."""
        super(TestHPFScoringMethods, self).__init__(*args, **kwargs)
        self.local_obj = LocalDataStore("tests/test_data")
        self.hpf_obj = HPFScoring(self.local_obj)
        self.hpf_obj_feedback = HPFScoring(self.local_obj, USE_FEEDBACK="True")

    def test_basic_object(self):
        """Test basic HPF object."""
        assert self.hpf_obj is not None
        assert self.hpf_obj.theta is not None
        assert self.hpf_obj.beta is not None
        assert self.hpf_obj.alpha is None
        assert self.hpf_obj_feedback.alpha is not None

    def test_match_feedback_manifest(self):
        """Test match feedback manifest with dummy ids."""
        input_id_set = {1}
        id_ = self.hpf_obj_feedback.match_feedback_manifest(input_id_set)
        assert int(id_) == -1
        input_id_set = {64, 200, 66, 44}
        id_ = self.hpf_obj_feedback.match_feedback_manifest(input_id_set)
        assert int(id_) == 0
        id_ = self.hpf_obj.match_feedback_manifest(input_id_set)
        assert int(id_) == -1

    def test_recommend_known_user(self):
        recommendations = self.hpf_obj.recommend_known_user(
            0, [0])
        self.assertTrue(recommendations)

    def test_recommend_new_user(self):
        recommendation = self.hpf_obj.recommend_new_user([0], k=13)
        self.assertTrue(recommendation)

    def test_predict_missing(self):
        with app.app.app_context():
            recommendation = self.hpf_obj.predict(['missing-pkg'])
            self.assertFalse(recommendation[0])
            self.assertTrue(recommendation[2])

    def test_model_details(self):
        """Test the basic model details function."""
        details = """The model will be scored against
        12405 Packages,
        9523 Manifests,
        Theta matrix of size 0.9446182250976562 MB, and
        Beta matrix of size 1.2304611206054688 MB."""
        assert self.hpf_obj.model_details() == details
        assert self.hpf_obj_feedback.model_details() == details

    def test_get_sizeof(self):
        """Test static _getsizeof method."""
        int_value = 1
        int_size = 2.6702880859375e-05
        assert HPFScoring._getsizeof(int_value) == "{} MB".format(int_size)


if __name__ == '__main__':
    unittest.main()
