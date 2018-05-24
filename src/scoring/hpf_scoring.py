import numpy as np
import os
from edward.models import Poisson
from edward.models import Gamma
import tensorflow as tf
from collections import defaultdict
from src.config import(SCORING_THRESHOLD,
                       HPF_SCORING_REGION,
                       HPF_output_package_id_dict,
                       HPF_output_manifest_id_dict,
                       HPF_output_rating_matrix,
                       HPF_output_item_matrix,
                       a, a_c, c, c_c,
                       b_c, d_c, K)


class HPFScoring:

    def __init__(self, datastore=None, scoring_threshold=SCORING_THRESHOLD,
                 scoring_region=HPF_SCORING_REGION):
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
        self.rating_mean = 0.0
        self.a = tf.constant(a)
        self.a_c = tf.constant(a_c)
        self.c = tf.constant(c)
        self.c_c = tf.constant(c_c)
        self.b_c = tf.constant(b_c)
        self.d_c = tf.constant(d_c)
        self.K = K
        self.load()

    def load(self):
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
            rating_matrix_filename, "/tmp/rating_matrix.npy")
        self.rating_matrix = np.load("/tmp/rating_matrix.npy")
        beta_matrix_filename = os.path.join(
            self.scoring_region, HPF_output_item_matrix)
        self.datastore.download_file(
            beta_matrix_filename, "/tmp/beta_matrix.npy")
        self.beta = np.load("/tmp/beta_matrix.npy")
        self.manifests, self.packages = self.rating_matrix.shape
        self.rating_mean = self.rating_matrix.mean()

    def predict(self, input_stack):
        input_id_set = set()
        missing_packages = []
        for package_name in input_stack:
            package_id = self.package_id_dict.get(package_name)
            if package_id is None:
                missing_packages.append(package_name)
        else:
            input_id_set.add(package_id)
        # TODO: Check for known-unknown ratio before recommending
        companion_recommendation, package_topic_dict = self.get_recommendation(
            input_id_set)
        return companion_recommendation, package_topic_dict, missing_packages

    def match_manifest(self, input_id_set):
        for manifest_id, dependency_set in self.manifest_id_dict.items():
            if dependency_set == input_id_set:
                break
        else:
            manifest_id = -1
        return manifest_id

    def get_recommendation(self, input_id_set):
        manifest_id = int(self.match_manifest(input_id_set))
        if manifest_id == -1:
            theta = []
            with tf.Session() as sess:
                epsilon = Gamma(self.a_c, self.a_c / self.b_c).eval()
                theta_func = Gamma(self.a, epsilon).eval()
                theta.append(epsilon * theta_func)
                theta = np.array(theta * self.K)
                result = Poisson(np.dot(theta, np.transpose(self.beta))).eval()
            result = self.normalize_result(result)
        else:
            result = self.rating_matrix[manifest_id]
        return self.filter_recommendation(result)

    def normalize_result(self, result):
        maxn = result.max()
        min_max = maxn - result.min()
        for i in range(self.packages):
            result[i] = (maxn - result[i]) / min_max
        return result

    def filter_recommendation(self, result):
        recommendation = list(result.nonzero()[0])
        all_companion = {}
        companion_recommendation = []
        package_topic_dict = {}
        for package_id in recommendation:
            prob_score = result[package_id]
            if prob_score > self.scoring_threshold:
                package_name = self.id_package_dict[package_id]
                all_companion[package_name] = prob_score
        sorted_result = sorted(all_companion.items(), key=lambda x: x[
                               1], reverse=True)[:10]
        for each_result in sorted_result:
            recommendation = {
                "cooccurrence_probability": each_result[1] * 100,
                "package_name": each_result[0],
                "topic_list": []
            }
            package_topic_dict[each_result[0]] = []
            companion_recommendation.append(recommendation)
        return companion_recommendation, package_topic_dict
