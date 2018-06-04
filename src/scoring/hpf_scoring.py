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
        self.K = K
        self.epsilon = Gamma(tf.constant(
            a_c), tf.constant(a_c) / tf.constant(b_c))
        self.theta_func = Gamma(tf.constant(a), self.epsilon)
        self.loadS3()

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
            if package_id is None:
                missing_packages.append(package_name)
        else:
            input_id_set.add(package_id)
            package_topic_dict[package_name] = []
        # TODO: Check for known-unknown ratio before recommending
        companion_recommendation = self.folding_in(
            input_id_set)
        return companion_recommendation, package_topic_dict, missing_packages

    def match_manifest(self, input_id_set):
        """Find a manifest list that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched manifest.
        """
        for manifest_id, dependency_set in self.manifest_id_dict.items():
            if dependency_set == input_id_set:
                break
        else:
            manifest_id = -1
        return manifest_id

    def folding_in(self, input_id_set):
        """Folding in logic for prediction.

        :param  input_id_set: A set containing package ids of user's input package list.
        :return: Filter companion recommendations and their topics.
        """
        manifest_id = int(self.match_manifest(input_id_set))
        if manifest_id == -1:
            theta = []
            with tf.Session() as sess:
                sess.run(self.theta_func)
                theta.append(self.epsilon.eval() * self.theta_func.eval())
                theta = np.array(theta * self.K)
                result = Poisson(np.dot(theta, np.transpose(self.beta))).eval()
            result = self.normalize_result(result)
        else:
            result = self.rating_matrix[manifest_id]
        return self.filter_recommendation(result)

    def normalize_result(self, result):
        """Normalise the probability score of the resulting recommendation.

        :param result: The Unnormalised recommendation result array.
        :return result: The normalised recommendation result array.
        """
        maxn = result.max()
        min_max = maxn - result.min()
        for i in range(self.packages):
            result[i] = (maxn - result[i]) / min_max
        return result

    def filter_recommendation(self, result):
        """Filter companion recommendations based on sorted threshold score.

        :param result: The unfiltered companion recommendation result.
        :return companion_recommendation: The filtered list of recommended companion packges
        along with condifence score.
        :return package_topic_dict: The topics associated with the packages
        in the input_stack+recommendation.
        """
        highest_indices = result.argsort()[-11:-1]
        companion_recommendation = []
        for package_id in highest_indices:
            prob_score = result[package_id]
            package_name = self.id_package_dict[package_id]
            recommendation = {
                "cooccurrence_probability": prob_score * 100,
                "package_name": package_name,
                "topic_list": []
            }
            companion_recommendation.append(recommendation)
        return companion_recommendation
