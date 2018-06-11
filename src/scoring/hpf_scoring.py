"""The HPF Model scoring class."""

import numpy as np
from scipy import sparse
import os
from edward.models import Poisson
from edward.models import Gamma
import tensorflow as tf
from collections import defaultdict
from src.config import (SCORING_THRESHOLD,
                        HPF_SCORING_REGION,
                        HPF_output_package_id_dict,
                        HPF_output_manifest_id_dict,
                        HPF_output_rating_matrix,
                        HPF_output_item_matrix,
                        a, a_c, c, c_c,
                        b_c, d_c, K)


class HPFScoring:
    """The HPF Model scoring class."""

    def __init__(self, datastore=None, scoring_threshold=SCORING_THRESHOLD,
                 scoring_region=HPF_SCORING_REGION):
        """Set the variables and load model data."""
        self.datastore = datastore
        self.scoring_threshold = scoring_threshold
        self.scoring_region = scoring_region
        self.package_id_dict = None
        self.id_package_dict = None
        self.rating_matrix = None
        self.beta = None
        self.manifest_id_dict = None
        self.manifests = 0
        self.packages = 0
        self.sess = tf.Session()
        self.epsilon = Gamma(tf.constant(
            a_c), tf.constant(a_c) / tf.constant(b_c)).eval(session=self.sess)
        self.theta = np.array([self.epsilon * Gamma(tf.constant(
            a), self.epsilon).eval(session=self.sess)] * K)
        self.loadS3()
        self.dummy_result = Poisson(
            np.dot(self.theta, np.transpose(self.beta))).eval(session=self.sess)
        self.normalize_result()

    def loadS3(self):
        """Load the model data from AWS S3."""
        package_id_dict_filename = os.path.join(
            self.scoring_region, HPF_output_package_id_dict)
        self.package_id_dict = self.datastore.read_json_file(
            package_id_dict_filename)
        self.id_package_dict = {x: n for n, x in self.package_id_dict.items()}
        manifest_id_dict_filename = os.path.join(
            self.scoring_region, HPF_output_manifest_id_dict)
        self.manifest_id_dict = self.datastore.read_json_file(
            manifest_id_dict_filename)
        self.manifest_id_dict = {n: set(x)
                                 for n, x in self.manifest_id_dict.items()}
        rating_matrix_filename = os.path.join(
            self.scoring_region, HPF_output_rating_matrix)
        self.datastore.download_file(
            rating_matrix_filename, "/tmp/rating_matrix.npz")
        sparse_matrix = sparse.load_npz('/tmp/rating_matrix.npz')
        self.rating_matrix = sparse_matrix.toarray()
        del(sparse_matrix)
        beta_matrix_filename = os.path.join(
            self.scoring_region, HPF_output_item_matrix)
        self.datastore.download_file(
            beta_matrix_filename, "/tmp/item_matrix.npz")
        sparse_matrix = sparse.load_npz('/tmp/item_matrix.npz')
        self.beta = sparse_matrix.toarray()
        del(sparse_matrix)
        self.manifests, self.packages = self.rating_matrix.shape

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
        input_id_set = set()
        missing_packages = []
        package_topic_dict = {}
        for package_name in input_stack:
            package_id = self.package_id_dict.get(package_name)
            if package_id:
                input_id_set.add(package_id)
                package_topic_dict[package_name] = []
            else:
                missing_packages.append(package_name)

        # TODO: Check for known-unknown ratio before recommending
        companion_recommendation = self.folding_in(
            input_id_set)
        return companion_recommendation, package_topic_dict, missing_packages

    def match_manifest(self, input_id_set):
        """Find a manifest list that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched manifest.
        """
        manifest_id = -1
        for manifest_id, dependency_set in self.manifest_id_dict.items():
            if dependency_set == input_id_set:
                break
        return manifest_id

    def folding_in(self, input_id_set):
        """Folding in logic for prediction.

        :param  input_id_set: A set containing package ids of user's input package list.
        :return: Filter companion recommendations and their topics.
        """
        manifest_id = int(self.match_manifest(input_id_set))
        if manifest_id == -1:
            result = np.array(self.dummy_result)
        else:
            result = self.rating_matrix[manifest_id]
        return self.filter_recommendation(result)

    def normalize_result(self):
        """Normalise the probability score of the resulting recommendation.

        :param result: The Unnormalised recommendation result array.
        :return result: The normalised recommendation result array.
        """
        maxn = self.dummy_result.max()
        min_max = maxn - self.dummy_result.min()
        for i in range(self.packages):
            value = 0
            try:
                value = (maxn - self.dummy_result[i]) / min_max
            except Exception as e:
                print(e)
            finally:
                self.dummy_result[i] = value

    def filter_recommendation(self, result):
        """Filter companion recommendations based on sorted threshold score.

        :param result: The unfiltered companion recommendation result.
        :return companion_recommendation: The filtered list of recommended companion packges
        along with condifence score.
        :return package_topic_dict: The topics associated with the packages
        in the input_stack+recommendation.
        """
        # TODO: Add a param for variable len of highest_indices
        highest_indices = result.argsort()[-5:len(result)]
        companion_recommendation = []
        for package_id in highest_indices:
            recommendation = {
                "cooccurrence_probability": result[package_id] * 100,
                "package_name": self.id_package_dict[package_id],
                "topic_list": []
            }
            companion_recommendation.append(recommendation)
        return companion_recommendation
