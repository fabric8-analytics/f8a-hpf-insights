"""The HPF Model scoring class."""

import logging
import os
import itertools
from collections import OrderedDict
from sys import getsizeof
import pandas as pd
import numpy as np
from flask import current_app

from src.config import (HPF_SCORING_REGION, HPF_MODEL_PATH,
                        HPF_output_manifest_id_dict,
                        HPF_output_package_id_dict, MIN_REC_CONFIDENCE)

# To turn off tensorflow CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Turn off the annoying boto logs unless some error occurs
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('s3transfer').setLevel(logging.ERROR)
logging.getLogger('boto3').setLevel(logging.ERROR)

if current_app:  # pragma: no cover
    _logger = current_app.logger
else:
    _logger = logging.getLogger(__file__)
    _logger.setLevel(level=logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    _logger.addHandler(consoleHandler)


class HPFScoring:
    """The HPF Model scoring class."""

    def __init__(self, datastore=None, num_recommendations=5):
        """Set the variables and load model data."""
        self.datastore = datastore
        self.package_id_dict = OrderedDict()
        self.id_package_dict = OrderedDict()
        self.manifest_id_dict = OrderedDict()
        self.feedback_id_dict = OrderedDict()
        self.manifests = 0
        self.recommender = self._load_model()
        self.logger = logging.getLogger(__name__ + '.HPFScoring')
        self.packages = 0
        self.m = num_recommendations
        self.loadObjects()

    @staticmethod
    def _getsizeof(attribute):
        """Return the size of attribute in MBs.

        param attribute: The object's attribute.
        """
        return "{} MB".format(getsizeof(attribute) / 1024 / 1024)

    def _load_model(self):
        """Load the model from s3."""
        return self.datastore.read_pickle_file(HPF_MODEL_PATH)

    def model_details(self):
        """Return the model details size."""
        details = """The model will be scored against
        {} Packages,
        {} Manifests""". \
            format(
                len(self.package_id_dict),
                len(self.manifest_id_dict))
        return details

    def loadObjects(self):  # pragma: no cover
        """Load the model data from AWS S3."""
        package_id_dict_filename = os.path.join(HPF_SCORING_REGION, HPF_output_package_id_dict)
        self.package_id_dict = self.datastore.read_json_file(package_id_dict_filename)
        self.id_package_dict = OrderedDict({x: n for n, x in self.package_id_dict[
            0].get("package_list", {}).items()})
        self.package_id_dict = OrderedDict(self.package_id_dict[0].get("package_list", {}))

        manifest_id_dict_filename = os.path.join(HPF_SCORING_REGION, HPF_output_manifest_id_dict)
        self.manifest_id_dict = self.datastore.read_json_file(manifest_id_dict_filename)
        self.manifest_id_dict = OrderedDict({n: set(x) for n, x in self.manifest_id_dict[
            0].get("manifest_list", {}).items()})

    def predict(self, input_stack):
        """Prediction function.

        :param input_stack: The user's package list
        for which companion recommendation are to be generated.
        :return companion_recommendation: The list of recommended companion packages
        along with condifence score.
        :return package_topic_dict: The topics associated with the packages
        in the input_stack+recommendation.
        :return missing_packages: The list of packages unknown to the HPF model.
        """
        input_stack = set(input_stack)
        input_id_set = set()
        missing_packages = set()
        package_topic_dict = {}
        companion_recommendation = []
        if not input_stack:
            return companion_recommendation, package_topic_dict, list(missing_packages)
        for package_name in input_stack:
            package_id = self.package_id_dict.get(package_name)
            if package_id:
                input_id_set.add(package_id)
                package_topic_dict[package_name] = []
            else:
                missing_packages.add(package_name)

        if len(input_stack) - len(missing_packages) == 0 or \
                len(missing_packages) > len(input_stack) - len(missing_packages):
            current_app.logger.info("Not recommending as stack has {} missing packages out "
                                    "of {}".format(len(missing_packages), len(input_stack)))
            return [], {}, list(missing_packages)
        manifest_match = self.match_manifest(input_id_set)
        if manifest_match > 0:
            result, user_id = self.recommend_known_user(manifest_match)
        else:
            result, user_id = self.recommend_new_user(list(input_id_set))
        companion_recommendation = self.filter_recommendation(result, input_id_set, user_id)
        return companion_recommendation, package_topic_dict, list(missing_packages)

    def match_manifest(self, input_id_set):  # pragma: no cover
        """Find a manifest list that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched manifest.
        """
        # no blank line allowed here

        # TODO: Go back to the difference based logic, this simpler logic will do for now.

        for manifest_id, dependency_set in self.manifest_id_dict.items():
            if input_id_set.issubset(dependency_set):
                current_app.logger.debug(
                    "input_id_set {} and manifest_id {}".format(input_id_set, manifest_id))
                return int(manifest_id)
        return -1

    def package_labelling(self, package_list):
        """Will return package names for given package id list."""
        labeled_packages = self.id_package_dict
        labeled_package_list = [labeled_packages[i] for i in package_list]
        return labeled_package_list

    def recommend_known_user(self, user_match):
        """Give the recommendation for a user(manifest) that was in the training set."""
        _logger.debug("Recommending for existing user: {}".format(user_match))
        recommendations = self.recommender.topN(user=user_match, n=self.m)

        return recommendations, user_match

    def recommend_new_user(self, input_id_set):
        """
        Implement the 'add_user' addon of HPFREC model.

        For more information please follow:
        https://hpfrec.readthedocs.io/en/latest/source/hpfrec.html
        """
        # no blank line allowed here
        new_df = pd.DataFrame({
            'ItemId': input_id_set,
            'Count': [1] * len(input_id_set)})
        user_id = self.recommender.nusers
        is_user_added = self.recommender.add_user(user_id=user_id, counts_df=new_df)
        if is_user_added:
            user_id -= 1
            recommendations = self.recommender.topN(user_id, n=self.m)
            return recommendations, user_id

        return _logger.info('Unable to add user')

    def filter_recommendation(self, result, input_stack, user_id):
        """Use for co-occurrence probability and for filtering of companion packages."""
        package_id_set = input_stack
        recommendations = result
        companion_packages = []
        recommendations = \
            np.array(list(itertools.compress(recommendations,
                                             [i not in package_id_set for i in recommendations])))

        print("Filtered recommendation ids are: " + str(recommendations))

        poisson_values = self.recommender.predict(
            user=[user_id] * recommendations.size,
            item=recommendations
        )

        def sigmoid(array):
            return 1 / (1 + np.exp(-array))

        # This is copy pasted on as is basis from Pypi and NPM model.
        # It's not the right way of calculating probability by any means.
        # There is a more lengthier way to calculate probabilities using
        # logistic regression which remains to be implemented
        # (but that's also not completely correct).
        # For discussion please follow: https://github.com/david-cortes/hpfrec/issues/4
        normalized_poisson_values = sigmoid(
            (poisson_values - poisson_values.mean()) / poisson_values.std()) * 100

        filtered_packages = self.package_labelling(recommendations)

        for idx, package in enumerate(filtered_packages):
            if normalized_poisson_values[idx] >= MIN_REC_CONFIDENCE:
                companion_packages.append({
                    "package_name": package,
                    "cooccurrence_probability": str(normalized_poisson_values[idx]),
                    "topic_list": []
                })

        return companion_packages
