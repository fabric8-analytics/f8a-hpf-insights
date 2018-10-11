"""Generate the feedback matrix.

Logic: https://github.com/fabric8-analytics/f8a-hpf-insights/wiki/Feedback-Logic
If auth is to incorporated, update the DockerFiles using License Service as example.
"""

import requests
from collections import OrderedDict
from scipy import sparse
import numpy as np
import logging
import os

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from src.config import (HPF_SCORING_REGION,
                        HPF_output_package_id_dict,
                        feedback_threshold,
                        HPF_output_feedback_matrix,
                        HPF_output_feedback_id_dict)
# Import this for standalone testing of feedback matrix, requires a pre-generated package_id_dict.
from src.data_store.local_data_store import LocalDataStore


class Feedback:
    """Generate and save feedback."""

    def __init__(self, datastore=None):
        """Set feedback varibales."""
        self.datastore = datastore
        self.feedback_raw_json = []
        # TODO: Improve selection of feedback_type
        # For only alternate, feedback_type=set(["alternate"])
        # For both, feedback_type=set(["alternate", "companion"])
        self.feedback_type = set(["companion"])
        self.package_id_dict = {}
        # TODO: for traning and scoring, use a ordered feedback_id_dict.
        self.feedback_id_dict = {}
        self.feedback_json_clean = {}
        self.alpha_matrix = None
        self.normalization_dict = {}

    def load_package_dict(self):
        """Load the previously saved package dict."""
        # TODO: add a try catch,
        # and save empty feedback_id_dict and empty alpha_matrix if error occurs while fetching.
        package_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_package_id_dict)
        self.package_id_dict = self.datastore.read_json_file(
            package_id_dict_filename)
        self.package_id_dict = OrderedDict(
            self.package_id_dict[0].get("package_list", {}))

    def get_raw_feedback_json(self):
        """Get the raw feedback data from POSTGRES."""
        URL_postfix = "api/v1/recommendation_feedback/" + HPF_SCORING_REGION
        URL_prefix_stage = "https://recommender.api.prod-preview.openshift.io/"
        # URL_prefix_prod = "https://recommender.api.openshift.io/"
        # Todo: Be able to use stage URL on stage and prod URL on prod
        URL = URL_prefix_stage + URL_postfix
        # TODO: add a try catch, and
        # save empty feedback_id_dict and empty alpha_matrix if error occurs.
        r = requests.get(url=URL)
        self.feedback_raw_json = r.json()

    def generate_feedback_id_dict(self):
        """Generate the feedback manifests id dict."""
        count = 0
        for each_feedback in self.feedback_raw_json:
            package_set = set()
            manifest = [each_pck.get("package", "")
                        for each_pck in each_feedback.get("input_package_list", [])]
            # Todo add a case of returned package_id is -1
            for each_package in manifest:
                package_set.add(self.package_id_dict[each_package])
            self.feedback_id_dict[count] = list(package_set)
            count += 1

    def match_feedback_manifest(self, input_id_set):
        """Find a feedback manifest that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched feedback manifest.
        """
        for manifest_id, dependency_set in self.feedback_id_dict.items():
            if set(dependency_set) == input_id_set:
                break
        else:
            manifest_id = -1
        logger.debug(
            "input_id_set {} and feedback_manifest_id {}".format(input_id_set, manifest_id))
        return int(manifest_id)

    def generate_clean_feedback_json(self):
        """Process the raw feedback json, and generate the feedback json.

        It done for all unique manifest and the unique packages recommended for that manifest.
        This acts as the intermediate feedback data to be converted to the feedback matrix
        """
        for each_feedback in self.feedback_raw_json:
            if not each_feedback.get("recommendation_type") in self.feedback_type:
                continue
            package_set = set()
            positive = 0
            negative = 0
            manifest = [each_pck.get("package", "")
                        for each_pck in each_feedback.get("input_package_list", [])]
            for each_package in manifest:
                package_set.add(self.package_id_dict[each_package])
            manifest_id = self.match_feedback_manifest(package_set)
            # This condition should not occur, but has been put for sanity.
            if manifest_id == -1:
                continue
            recommended_package = each_feedback.get("recommended_package_name", "")
            recommended_package = int(self.package_id_dict.get(recommended_package, -1))
            # This condition should not occur, but has been put for sanity.
            if recommended_package == -1:
                continue
            """
            3 cases can occur while filling the feedback json dict:
            CASE 1: Manifest exits in the feedback_json.
            CASE 2: Manifest exists but not the recommended package.
            CASE 3: Both manifest and recommended package exist.
            """
            # TODO: Improve the cyclomatic complexity of the 3 cased nested if-else
            if manifest_id in self.feedback_json_clean and\
               recommended_package in self.feedback_json_clean[manifest_id]:
                positive = self.feedback_json_clean[manifest_id][recommended_package]["positive"]
                negative = self.feedback_json_clean[manifest_id][recommended_package]["negative"]
                if each_feedback.get("feedback", False):
                    positive += 1
                else:
                    negative += 1
                self.feedback_json_clean[manifest_id][recommended_package]["positive"] = positive
                self.feedback_json_clean[manifest_id][recommended_package]["negative"] = negative
            elif manifest_id in self.feedback_json_clean:
                if each_feedback.get("feedback", False):
                    positive = 1
                else:
                    negative = 1
                rec_dict = {recommended_package: {"positive": positive, "negative": negative}}
                self.feedback_json_clean[manifest_id].update(rec_dict)
            else:
                self.feedback_json_clean[manifest_id] = {}
                if each_feedback.get("feedback", False):
                    positive = 1
                else:
                    negative = 1
                rec_dict = {recommended_package: {"positive": positive, "negative": negative}}
                self.feedback_json_clean[manifest_id] = rec_dict

    def normalize_feedback_manifest(self):
        """Generate the normlaization dict.

        It is later used to perform the user normalization step.
        Normalise feedback score based on the number of users who gave feedback for that manifest.
        """
        for manifest_id in self.feedback_json_clean:
            max_num_users = 0
            min_num_users = 0
            for recommended_package in self.feedback_json_clean[manifest_id]:
                p = self.feedback_json_clean[manifest_id][recommended_package]["positive"]
                n = self.feedback_json_clean[manifest_id][recommended_package]["negative"]
                total_feedback = p + n
                if total_feedback < min_num_users:
                    min_num_users = total_feedback
                elif total_feedback > max_num_users:
                    max_num_users = total_feedback
            self.normalization_dict[manifest_id] = (min_num_users, max_num_users)

    def generate_normalised_feedback_matrix(self):
        """Generate the feedback matrix aka alpha matrix, and normalise it."""
        manifests = len(self.feedback_id_dict)
        packages = len(self.package_id_dict)
        self.alpha_matrix = self.rating_matrix = np.full((manifests, packages), feedback_threshold)
        for manifest_id in self.feedback_json_clean:
            min_user, max_user = self.normalization_dict[manifest_id]
            min_max = max_user - min_user
            for recommended_package in self.feedback_json_clean[manifest_id]:
                p = self.feedback_json_clean[manifest_id][recommended_package]["positive"]
                n = self.feedback_json_clean[manifest_id][recommended_package]["negative"]
                total_feedback = p + n
                # Means that this package was never recommeded for this manifest.
                if total_feedback == 0:
                    self.alpha_matrix[manifest_id][recommended_package] = feedback_threshold
                else:
                    positive_rating = p / total_feedback
                    user_normalization = 1
                    if min_max != 0:
                        user_normalization = total_feedback / min_max
                    self.alpha_matrix[manifest_id][recommended_package] = positive_rating * \
                        user_normalization

    def save_feedback_matrix(self):
        """Save the feedback_id_dict and feedback_alpha matrix to S3."""
        feedback_id_dict_list = []
        feedback_id_dict_list.append(
            {"ecosystem": HPF_SCORING_REGION, "feedback_list": self.feedback_id_dict})
        feedback_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_feedback_id_dict)
        self.datastore.write_json_file(
            feedback_id_dict_filename, feedback_id_dict_list)
        sparse_feedback_matrix = sparse.csr_matrix(self.alpha_matrix)
        sparse.save_npz('/tmp/hpf/feedback_matrix.npz', sparse_feedback_matrix)
        # Extra upload to save on S3/minio
        if not isinstance(self.datastore, LocalDataStore):
            feedback_matrix_filename = os.path.join(
                HPF_SCORING_REGION, HPF_output_feedback_matrix)
            self.datastore.upload_file(
                "/tmp/hpf/feedback_matrix.npz", feedback_matrix_filename)

    def execute(self):
        """Execute generation of feedback matrix."""
        self.get_raw_feedback_json()
        self.load_package_dict()
        self.generate_feedback_id_dict()
        self.generate_clean_feedback_json()
        self.normalize_feedback_manifest()
        self.generate_normalised_feedback_matrix()
        self.save_feedback_matrix()


# To test the feedback_script standalone. Add your own local datasource.
# if __name__ == '__main__':
#   local_obj=LocalDataStore("tests/test_data")
#   fb = Feedback(datastore=local_obj)
#   fb.execute()
