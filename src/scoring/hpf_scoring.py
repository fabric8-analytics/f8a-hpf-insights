"""The HPF Model scoring class."""

import numpy as np
from scipy import sparse
import os
from sys import getsizeof, maxsize
from collections import OrderedDict
from edward.models import Poisson
from edward.models import Gamma
import tensorflow as tf
import os
from flask import current_app
import logging
from collections import defaultdict
from src.data_store.s3_data_store import S3DataStore
from src.data_store.local_data_store import LocalDataStore
from src.utils import convert_string2bool_env
from src.config import (UNKNOWN_PACKAGES_THRESHOLD,
                        MAX_COMPANION_REC_COUNT,
                        HPF_SCORING_REGION,
                        HPF_output_package_id_dict,
                        HPF_output_manifest_id_dict,
                        HPF_output_feedback_id_dict,
                        HPF_output_user_matrix,
                        HPF_output_item_matrix,
                        HPF_output_feedback_matrix,
                        a, a_c, c, c_c,
                        b_c, d_c, K,
                        USE_FEEDBACK,
                        feedback_threshold)

# To turn off tensorflow CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if current_app:  # pragma: no cover
    _logger = current_app.logger
else:
    _logger = logging.getLogger(__file__)
    _logger.setLevel(level=logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    _logger.addHandler(consoleHandler)


class HPFScoring:
    """The HPF Model scoring class."""

    def __init__(self, datastore=None, USE_FEEDBACK=USE_FEEDBACK):
        """Set the variables and load model data."""
        self.datastore = datastore
        self.USE_FEEDBACK = convert_string2bool_env(USE_FEEDBACK)
        self.package_id_dict = OrderedDict()
        self.id_package_dict = OrderedDict()
        self.beta = None
        self.theta = None
        self.alpha = None
        self.manifest_id_dict = OrderedDict()
        self.feedback_id_dict = OrderedDict()
        self.manifests = 0
        self.packages = 0
        self.epsilon = Gamma(tf.constant(
            a_c), tf.constant(a_c) / tf.constant(b_c)).\
            prob(tf.constant(K, dtype=tf.float32)).eval(session=tf.Session())
        self.theta_dummy = Poisson(np.array([self.epsilon * Gamma(tf.constant(
            a), self.epsilon).prob(tf.constant(K, dtype=tf.float32)).
            eval(session=tf.Session())] * K, dtype=float))
        if isinstance(datastore, S3DataStore):  # pragma: no-cover
            self.load_s3()
        else:
            self.load_local()
        self.manifests = self.theta.shape[0]
        self.packages = self.beta.shape[0]
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
        details = """The model will be scored against
        {} Packages,
        {} Manifests,
        Theta matrix of size {}, and
        Beta matrix of size {}.""".\
            format(
                len(self.package_id_dict),
                len(self.manifest_id_dict),
                HPFScoring._getsizeof(self.theta),
                HPFScoring._getsizeof(self.beta))
        return details

    def load_s3(self):  # pragma: no cover
        """Load the model data from AWS S3."""
        theta_matrix_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_user_matrix)
        self.datastore.download_file(
            theta_matrix_filename, "/tmp/user_matrix.npz")
        sparse_matrix = sparse.load_npz('/tmp/user_matrix.npz')
        self.theta = sparse_matrix.toarray()
        del(sparse_matrix)
        os.remove("/tmp/user_matrix.npz")
        beta_matrix_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_item_matrix)
        self.datastore.download_file(
            beta_matrix_filename, "/tmp/item_matrix.npz")
        sparse_matrix = sparse.load_npz('/tmp/item_matrix.npz')
        self.beta = sparse_matrix.toarray()
        del(sparse_matrix)
        os.remove("/tmp/item_matrix.npz")
        if self.USE_FEEDBACK:
            alpha_matrix_filename = os.path.join(
                HPF_SCORING_REGION, HPF_output_feedback_matrix)
            self.datastore.download_file(
                alpha_matrix_filename, "/tmp/alpha_matrix.npz")
            sparse_matrix = sparse.load_npz("/tmp/alpha_matrix.npz")
            self.alpha = sparse_matrix.toarray()
            del(sparse_matrix)
        self.load_jsons()

    def load_local(self):
        """Load the model data from AWS S3."""
        theta_matrix_filename = os.path.join(
            self.datastore.src_dir, HPF_SCORING_REGION, HPF_output_user_matrix)
        sparse_matrix = sparse.load_npz(theta_matrix_filename)
        self.theta = sparse_matrix.toarray()
        del(sparse_matrix)
        beta_matrix_filename = os.path.join(
            self.datastore.src_dir,
            HPF_SCORING_REGION, HPF_output_item_matrix)
        sparse_matrix = sparse.load_npz(beta_matrix_filename)
        self.beta = sparse_matrix.toarray()
        del(sparse_matrix)
        if self.USE_FEEDBACK:
            alpha_matrix_filename = os.path.join(
                self.datastore.src_dir,
                HPF_SCORING_REGION, HPF_output_feedback_matrix)
            sparse_matrix = sparse.load_npz(alpha_matrix_filename)
            self.alpha = sparse_matrix.toarray()
            del(sparse_matrix)
        self.load_jsons()

    def load_jsons(self):
        """Load Json files via common methods for S3 and local."""
        package_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_package_id_dict)
        self.package_id_dict = self.datastore.read_json_file(
            package_id_dict_filename)
        self.id_package_dict = OrderedDict({x: n for n, x in self.package_id_dict[
            0].get("package_list", {}).items()})
        self.package_id_dict = OrderedDict(
            self.package_id_dict[0].get("package_list", {}))
        manifest_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_manifest_id_dict)
        self.manifest_id_dict = self.datastore.read_json_file(
            manifest_id_dict_filename)
        self.manifest_id_dict = OrderedDict({n: set(x) for n, x in self.manifest_id_dict[
            0].get("manifest_list", {}).items()})
        if self.USE_FEEDBACK:
            feedback_id_dict_filename = os.path.join(
                HPF_SCORING_REGION, HPF_output_feedback_id_dict)
            self.feedback_id_dict = self.datastore.read_json_file(
                feedback_id_dict_filename)
            self.feedback_id_dict = OrderedDict({n: set(x) for n, x in self.feedback_id_dict[
                0].get("feedback_list", {}).items()})

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
        if len(missing_packages) / len(input_stack) < UNKNOWN_PACKAGES_THRESHOLD:
            companion_recommendation = self.folding_in(
                input_id_set)
        else:
            _logger.error(
                "{} length of missing packages beyond unknow threshold value of {}".format(
                    len(missing_packages), UNKNOWN_PACKAGES_THRESHOLD))
        return companion_recommendation, package_topic_dict, list(missing_packages)

    def match_manifest(self, input_id_set):  # pragma: no cover
        """Find a manifest list that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched manifest.
        """
        closest_manifest_id = -1
        max_diff = maxsize
        for manifest_id, dependency_set in self.manifest_id_dict.items():
            curr_diff = len(dependency_set.difference(input_id_set))
            if dependency_set == input_id_set:
                closest_manifest_id = manifest_id
                break
            elif input_id_set.issubset(dependency_set) and curr_diff < max_diff:
                closest_manifest_id = manifest_id
                max_diff = curr_diff
        _logger.debug(
            "input_id_set {} and manifest_id {}".format(input_id_set, closest_manifest_id))
        return closest_manifest_id

    def match_feedback_manifest(self, input_id_set):
        """Find a feedback manifest that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched feedback manifest.
        """
        for manifest_id, dependency_set in self.feedback_id_dict.items():
            if dependency_set == input_id_set:
                break
        else:
            manifest_id = -1
        _logger.debug(
            "input_id_set {} and feedback_manifest_id {}".format(input_id_set, manifest_id))
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
        normalised_result = self.normalize_result(result, input_id_set)
        if self.USE_FEEDBACK:
            alpha_id = int(self.match_feedback_manifest(input_id_set))
            return self.filter_recommendation(normalised_result,
                                              alpha_id=alpha_id)
        return self.filter_recommendation(normalised_result)

    def normalize_result(self, result, input_id_set, array_len=None):
        """Normalise the probability score of the resulting recommendation.

        :param result: The non-normalised recommendation result array.
        :param input_id_set: The user's input package ids.
        :param array_len: length of normalised result array.
        :return normalised_result: The normalised recommendation result array.
        """
        if array_len is None:
            array_len = self.packages
        normalised_result = np.array([-1.0 if i in input_id_set
                                      else result[i].mean()
                                      for i in range(array_len)])
        return normalised_result

    def filter_recommendation(self, result, alpha_id=-1, max_count=MAX_COMPANION_REC_COUNT):
        """Filter companion recommendations based on sorted threshold score.

        :param result: The unfiltered companion recommendation result.
        :param max_count: Maximum number of recommendations to return.
        :return companion_recommendation: The filtered list of recommended companion packages
        along with condifence score.
        :return package_topic_dict: The topics associated with the packages
        in the input_stack+recommendation.
        """
        highest_indices = set(result.argsort()[-max_count:len(result)])
        companion_recommendation = []
        if self.USE_FEEDBACK and alpha_id != -1:
            alpha_set = set(
                np.where(self.alpha[alpha_id] >= feedback_threshold)[0])
            highest_indices = highest_indices.intersection(alpha_set)

        for package_id in highest_indices:
            recommendation = {
                "cooccurrence_probability": result[package_id] * 100,
                "package_name": self.id_package_dict[package_id],
                "topic_list": []
            }
            companion_recommendation.append(recommendation)
        return companion_recommendation
