"""Test functionalities of hpf scoring."""
import unittest
from collections import OrderedDict
from src.flask_endpoint import app
from rudra.data_store.local_data_store import LocalDataStore
from src.scoring.hpf_scoring import HPFScoring
from src.config import (HPF_output_manifest_id_dict,
                        HPF_output_package_id_dict)


class TestHPFScoringMethods(unittest.TestCase):
    """Test functionalities of hpf scoring."""

    def __init__(self, *args, **kwargs):
        """Initialise the local data store and HPF object."""
        super(TestHPFScoringMethods, self).__init__(*args, **kwargs)
        self.local_obj = LocalDataStore("tests/test_data")
        self.hpf_obj = HPFScoring(self.local_obj)
        self.hpf_obj_feedback = HPFScoring(self.local_obj)
        self.package_id_dict = OrderedDict()
        self.id_package_dict = OrderedDict()
        self.manifest_id_dict = OrderedDict()

    def test_basic_object(self):
        """Test basic HPF object."""
        assert self.hpf_obj is not None
        assert self.hpf_obj.recommender is not None
        assert self.hpf_obj.m is not None

    # Currently we are not moving forward with this, but in future will look
    # on it. So commented.
    # def test_match_feedback_manifest(self):
    #     """Test match feedback manifest with dummy ids."""
    #     input_id_set = {1}
    #     id_ = self.hpf_obj_feedback.match_feedback_manifest(input_id_set)
    #     assert int(id_) == -1
    #     input_id_set = {64, 200, 66, 44}
    #     id_ = self.hpf_obj_feedback.match_feedback_manifest(input_id_set)
    #     assert int(id_) == 0
    #     id_ = self.hpf_obj.match_feedback_manifest(input_id_set)
    #     assert int(id_) == -1

    def test_load_objects(self):
        """Test logic where incoming data is correct or not."""
        self.package_id_dict = self.local_obj.read_json_file(HPF_output_package_id_dict)
        self.id_package_dict = OrderedDict({x: n for n, x in self.package_id_dict.get
                                            ("package_list", {}).items()})
        self.package_id_dict = OrderedDict(self.package_id_dict.get("package_list", {}))
        self.manifest_id_dict = self.local_obj.read_json_file(HPF_output_manifest_id_dict)
        self.manifest_id_dict = OrderedDict({n: set(x) for n, x in self.manifest_id_dict.get
                                            ("manifest_list", {}).items()})
        self.assertTrue(isinstance(self.package_id_dict, dict))
        self.assertTrue(isinstance(self.id_package_dict, dict))
        self.assertTrue(isinstance(self.manifest_id_dict, dict))

    def test_recommend_known_user(self):
        """Test logic where we recommend for a known user(exists in training set)."""
        recommendation, user_id = self.hpf_obj.recommend_known_user(
            0)
        assert recommendation is not None
        assert user_id is not None

    def test_recommend_new_user(self):
        """Test the fold-in logic where we calculate factors for new user."""
        recommendation, user_id = self.hpf_obj.recommend_new_user([0])
        assert recommendation is not None
        assert user_id is not None

    def test_predict_missing(self):
        """Test no prediction in case of higher than threshold missing package ratio."""
        with app.app.app_context():
            recommendation = self.hpf_obj.predict(['missing-pkg'])
            self.assertFalse(recommendation[0])
            self.assertTrue(recommendation[2])

    def test_model_details(self):
        """Test the basic model details function."""
        details = "The model will be scored against 12405 Packages, 9523 Manifests."

        print(self.hpf_obj.model_details())
        assert self.hpf_obj.model_details() == details
    #     # assert self.hpf_obj_feedback.model_details() == details

    def test_get_sizeof(self):
        """Test static _getsizeof method."""
        int_value = 1
        int_size = 2.6702880859375e-05
        assert HPFScoring._getsizeof(int_value) == "{} MB".format(int_size)


if __name__ == '__main__':
    unittest.main()
