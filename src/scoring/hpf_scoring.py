"""The HPF Model scoring class."""

import numpy as np
from scipy import sparse
import zipfile
import os
from sys import getsizeof
from edward.models import Poisson
from edward.models import Gamma
import tensorflow as tf
import os
from flask import current_app
from collections import defaultdict
from src.config import (UNKNOWN_PACKAGES_THRESHOLD,
                        MAX_COMPANION_REC_COUNT,
                        HPF_SCORING_REGION,
                        HPF_output_package_id_dict,
                        HPF_output_manifest_id_dict,
                        HPF_output_user_matrix,
                        HPF_output_item_matrix,
                        a, a_c, c, c_c,
                        b_c, d_c, K)

# To turn off tensorflow CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HPFScoring:
    """The HPF Model scoring class."""

    def __init__(self, datastore=None,
                 scoring_region=HPF_SCORING_REGION):
        """Set the variables and load model data."""
        self.datastore = datastore
        self.scoring_region = scoring_region
        self.package_id_dict = None
        self.id_package_dict = None
        self.beta = None
        self.theta = None
        self.manifest_id_dict = None
        self.manifests = 0
        self.packages = 0
        self.epsilon = Gamma(tf.constant(
            a_c), tf.constant(a_c) / tf.constant(b_c)).eval(session=tf.Session())
        self.theta_dummy = Poisson(np.array([self.epsilon * Gamma(tf.constant(
            a), self.epsilon).eval(session=tf.Session())] * K, dtype=float))
        self.loadS3()
        self.dummy_result = self.theta_dummy.prob(
            self.beta).eval(session=tf.Session())

    @staticmethod
    def _getsizeof(attribute):
        """Return the size of attribute in MBs.

        param attribute: The object's attribute.
        """
        return "{} MB".format(getsizeof(attribute) / 1024 / 1024)

    def model_details(self):
        """Return the model details size."""
        return(
            "The model will be scored against\
                {} Packages,\
                {} Manifests,\
                Theta matrix of size {}, and\
                Beta matrix of size {}.".format(
                len(self.package_id_dict),
                len(self.manifest_id_dict),
                HPFScoring._getsizeof(self.theta),
                HPFScoring._getsizeof(self.beta))
        )

    def loadS3(self):
        """Load the model data from AWS S3."""
        theta_matrix_filename = os.path.join(
            self.scoring_region, HPF_output_user_matrix)
        self.datastore.download_file(
            theta_matrix_filename, "/tmp/user_matrix.npz")
        sparse_matrix = sparse.load_npz('/tmp/user_matrix.npz')
        self.theta = sparse_matrix.toarray()
        del(sparse_matrix)
        os.remove("/tmp/user_matrix.npz")
        beta_matrix_filename = os.path.join(
            self.scoring_region, HPF_output_item_matrix)
        self.datastore.download_file(
            beta_matrix_filename, "/tmp/item_matrix.npz")
        sparse_matrix = sparse.load_npz('/tmp/item_matrix.npz')
        self.beta = sparse_matrix.toarray()
        del(sparse_matrix)
        os.remove("/tmp/item_matrix.npz")
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
        self.manifests = self.theta.shape[0]
        self.packages = self.beta.shape[0]

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
        for package_name in input_stack:
            package_id = self.package_id_dict.get(package_name)
            if package_id:
                input_id_set.add(package_id)
                package_topic_dict[package_name] = []
            else:
                missing_packages.add(package_name)
        if len(missing_packages) / len(input_stack) < UNKNOWN_PACKAGES_THRESHOLD:
            companion_recommendation = self.folding_in(
                input_id_set)
        else:
            current_app.logger.error(
                "{} length of missing packages beyond unknow threshold value of {}".format(
                    len(missing_packages), UNKNOWN_PACKAGES_THRESHOLD))
        return companion_recommendation, package_topic_dict, list(missing_packages)

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
        current_app.logger.debug(
            "input_id_set {} and manifest_id {}".format(input_id_set, manifest_id))
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
            graph_new = tf.Graph()
            with graph_new.as_default():
                result = Poisson(self.theta[manifest_id])
                result = result.prob(self.beta)
            with tf.Session(graph=graph_new) as sess_new:
                result = sess_new.run(result)
        normalised_result = self.normalize_result(result)
        return self.filter_recommendation(normalised_result)

    def normalize_result(self, result):
        """Normalise the probability score of the resulting recommendation.

        :param result: The Unnormalised recommendation result array.
        :return result: The normalised recommendation result array.
        """
        normalised_result = np.zeros([self.packages])
        for i in range(self.packages):
            normalised_result[i] = result[i].mean()
        return normalised_result

    def filter_recommendation(self, result):
        """Filter companion recommendations based on sorted threshold score.

        :param result: The unfiltered companion recommendation result.
        :return companion_recommendation: The filtered list of recommended companion packages
        along with condifence score.
        :return package_topic_dict: The topics associated with the packages
        in the input_stack+recommendation.
        """
        highest_indices = result.argsort()[-MAX_COMPANION_REC_COUNT:len(result)]
        companion_recommendation = []
        for package_id in highest_indices:
            recommendation = {
                "cooccurrence_probability": result[package_id] * 100,
                "package_name": self.id_package_dict[package_id],
                "topic_list": []
            }
            companion_recommendation.append(recommendation)
        return companion_recommendation
