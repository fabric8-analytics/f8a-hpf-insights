"""The HPF Model scoring class."""

import logging
import os
import time
from collections import OrderedDict
from math import exp, log
from sys import getsizeof

import numpy as np
import tensorflow as tf
from edward.models import Gamma, Uniform
from flask import current_app
from scipy import sparse
from scipy.special import psi

from src.config import (HPF_LAM_RTE_PATH, HPF_LAM_SHP_PATH, HPF_SCORING_REGION,
                        HPF_output_feedback_id_dict, HPF_output_feedback_matrix,
                        HPF_output_item_matrix, HPF_output_manifest_id_dict,
                        HPF_output_package_id_dict, HPF_output_user_matrix,
                        MAX_COMPANION_REC_COUNT, USE_FEEDBACK, a, a_c,
                        b_c, feedback_threshold, iter_score, stop_thr, MIN_REC_CONFIDENCE)
import src.config as config

from src.utils import convert_string2bool_env

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
        self.logger = logging.getLogger(__name__ + '.HPFScoring')
        self.packages = 0
        self.loadObjects()

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

    def loadObjects(self):  # pragma: no cover
        """Load the model data from AWS S3."""
        theta_matrix_filename = os.path.join(HPF_SCORING_REGION, HPF_output_user_matrix)
        self.datastore.download_file(theta_matrix_filename, "/tmp/hpf/user_matrix.npz")
        user_matrix_sparse = sparse.load_npz('/tmp/hpf/user_matrix.npz')
        self.theta = user_matrix_sparse.toarray()

        beta_matrix_filename = os.path.join(HPF_SCORING_REGION, HPF_output_item_matrix)
        self.datastore.download_file(beta_matrix_filename, "/tmp/hpf/item_matrix.npz")
        item_matrix_sparse = sparse.load_npz('/tmp/hpf/item_matrix.npz')
        self.beta = item_matrix_sparse.toarray()

        lam_shp_filename = os.path.join(HPF_SCORING_REGION, HPF_LAM_SHP_PATH)
        lam_rte_filename = os.path.join(HPF_SCORING_REGION, HPF_LAM_RTE_PATH)
        self.datastore.download_file(lam_shp_filename, "/tmp/hpf/lam_shp.npz")
        self.datastore.download_file(lam_rte_filename, "/tmp/hpf/lam_rte.npz")
        self.lam_shp = sparse.load_npz('/tmp/hpf/lam_shp.npz').toarray()
        self.lam_rte = sparse.load_npz('/tmp/hpf/lam_rte.npz').toarray()

        package_id_dict_filename = os.path.join(HPF_SCORING_REGION, HPF_output_package_id_dict)
        self.package_id_dict = self.datastore.read_json_file(package_id_dict_filename)
        self.id_package_dict = OrderedDict({x: n for n, x in self.package_id_dict[
            0].get("package_list", {}).items()})
        self.package_id_dict = OrderedDict(self.package_id_dict[0].get("package_list", {}))

        manifest_id_dict_filename = os.path.join(HPF_SCORING_REGION, HPF_output_manifest_id_dict)
        self.manifest_id_dict = self.datastore.read_json_file(manifest_id_dict_filename)
        self.manifest_id_dict = OrderedDict({n: set(x) for n, x in self.manifest_id_dict[
            0].get("manifest_list", {}).items()})
        self.manifests = self.theta.shape[0]
        self.packages = self.beta.shape[0]

        if self.USE_FEEDBACK:
            alpha_matrix_filename = os.path.join(HPF_SCORING_REGION, HPF_output_feedback_matrix)
            self.datastore.download_file(alpha_matrix_filename, "/tmp/hpf/feedback_matrix.npz")
            sparse_matrix = sparse.load_npz("/tmp/hpf/feedback_matrix.npz")
            self.alpha = sparse_matrix.toarray()
            feedback_id_dict_filename = os.path.join(
                HPF_SCORING_REGION, HPF_output_feedback_id_dict)
            self.feedback_id_dict = self.datastore.read_json_file(feedback_id_dict_filename)
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

        if len(input_stack) - len(missing_packages) == 0 or \
                len(missing_packages) > len(input_stack) - len(missing_packages):
            current_app.logger.info("Not recommending as stack has {} missing packages out "
                                    "of {}".format(len(missing_packages), len(input_stack)))
            return [], {}, list(missing_packages)
        manifest_match = self.match_manifest(input_id_set)
        if manifest_match > 0:
            companion_recommendation = self.recommend_known_user(manifest_match, input_id_set)
        else:
            companion_recommendation = self.recommend_new_user(list(input_id_set))
        return companion_recommendation, package_topic_dict, list(missing_packages)

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
        return int(manifest_id)

    def match_manifest(self, input_id_set):  # pragma: no cover
        """Find a manifest list that matches user's input package list and return its index.

        :param input_id_set: A set containing package ids of user's input package list.
        :return manifest_id: The index of the matched manifest.
        """
        # TODO: Go back to the difference based logic, this simpler logic will do for now.
        for manifest_id, dependency_set in self.manifest_id_dict.items():
            if input_id_set.issubset(dependency_set):
                current_app.logger.debug(
                        "input_id_set {} and manifest_id {}".format(input_id_set, manifest_id))
                return int(manifest_id)
        return -1

    def recommend_known_user(self, user_match, input_stack):
        """Give the recommendation for a user(manifest) that was in the training set."""
        _logger.debug("Recommending for existing user: {}".format(user_match))
        rec = np.dot(self.theta[user_match], self.beta.T)
        return self.filter_recommendation(rec, input_stack)

    def recommend_new_user(self, input_user_stack, k=config.K):
        """Implement the 'fold-in' logic.

        Calculates user factors for a new user and adds the user to the user matrix to make
        prediction.
        """
        # initializing parameters
        _logger.info("Could not find a match, calculating factors to recommend.")
        k_shp = a_c + k * a
        # TODO: t_shp, Why is it not required here?
        nY = len(input_user_stack)
        Y = np.ones(shape=(nY,))
        seed = np.random.seed(int(time.time()))
        theta = Gamma(a, 1 / b_c).sample(sample_shape=(k,), seed=seed).eval(session=tf.Session())
        k_rte = b_c + np.sum(theta)
        gamma_rte = \
            Gamma(a_c, b_c / a_c).sample(
                    sample_shape=(1,), seed=seed).eval(session=tf.Session()) + self.beta.sum(axis=0)
        gamma_shp = \
            gamma_rte * theta * \
            Uniform(low=.85, high=1.15).sample(
                sample_shape=(k,), seed=seed).eval(session=tf.Session())
        np.nan_to_num(gamma_shp, copy=False)
        np.nan_to_num(gamma_rte, copy=False)
        phi = np.empty((nY, k), dtype='float32')
        add_k_rte = a_c / b_c
        theta_prev = theta.copy()
        # running the loop
        for iter_num in range(iter_score):
            for i in range(nY):
                iid = input_user_stack[i]
                sumphi = 10e-6
                maxval = - 10 ** 1
                phi_st = i
                for j in range(k):
                    phi[phi_st, j] = psi(gamma_shp[j]) - log(gamma_rte[j]) + psi(
                            self.lam_shp[iid, j]) - log(self.lam_rte[iid, j])
                    if phi[phi_st, j] > maxval:
                        maxval = phi[phi_st, j]
                for j in range(k):
                    phi[phi_st, j] = exp(phi[phi_st, j] - maxval)
                    sumphi += phi[phi_st, j]
                for j in range(k):
                    phi[phi_st, j] *= Y[i] / sumphi
            gamma_rte = (k_shp / k_rte + self.beta.sum(axis=0, keepdims=True)).reshape(-1)
            gamma_shp = a + phi.sum(axis=0)
            theta = gamma_shp / gamma_rte
            k_rte = add_k_rte + theta.sum()
            # checking for early stop
            if np.linalg.norm(theta - theta_prev) < stop_thr:
                break
            theta_prev = theta.copy()
        rec = np.dot(theta, self.beta.T)
        return self.filter_recommendation(rec, input_user_stack)

    def filter_recommendation(self, result, input_stack):
        """Filter based on feedback, input set & max recommendation count.

        :param result: The unfiltered companion recommendation result.
        :return companion_recommendation: The filtered list of recommended companion packages
        along with condifence score.
        """
        companion_recomendations = []
        input_stack = list(input_stack)
        result = np.delete(result, input_stack)
        result = np.where(np.isfinite(result))[0]
        if self.USE_FEEDBACK:
            alpha_id = int(self.match_feedback_manifest(set(input_stack)))
            alpha_set = set(np.where(self.alpha[alpha_id] >= feedback_threshold)[0])
            result = np.delete(result, list(alpha_set))

        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))
        recommended_ids = result.argsort()[::-1][:MAX_COMPANION_REC_COUNT]
        mean = np.mean(result[recommended_ids])
        result = result - mean
        result = np.divide(result, np.std(result))
        for package_id in recommended_ids:
            recommendation = {
                "cooccurrence_probability": round(_sigmoid(result[package_id]) * 100, 2),
                "package_name": self.id_package_dict[package_id],
                "topic_list": []  # We don't have topics for this ecosystem!!
            }
            # At least thirty percent probability is required for recommendation to go through.
            if recommendation['cooccurrence_probability'] > MIN_REC_CONFIDENCE:
                companion_recomendations.append(recommendation)
        return companion_recomendations
