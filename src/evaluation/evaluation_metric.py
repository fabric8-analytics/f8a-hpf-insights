"""Perform evaluation."""

import numpy as np
import random
import tensorflow as tf
import time
import os
from scipy import sparse
from edward.models import Poisson
from edward.models import Gamma as IGR
from src.config import (HPF_SCORING_REGION,
                        HPF_output_user_matrix,
                        HPF_output_item_matrix,
                        HPF_evaluation,
                        K, MAX_COMPANION_REC_COUNT)
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EvaluationMetric:
    """Perform evaluation."""

    def __init__(self, datastore):
        """Perform evaluation, initialise matrices."""
        self.datastore = datastore
        self.theta = None
        self.beta = None
        self.test_data = None
        self.test_rows = []
        self.predict = None
        self.result = None
        self.time = time.gmtime()

    def generate_uid(self):
        """Generate a unique time string for storing result."""
        uid = ""
        fields = self.time.n_sequence_fields
        for i in range(fields):
            uid = uid + "_" + str(self.time[i])
        return uid + ".json"

    def rmse(self):
        """Return the root mean square error."""
        count = 0.0
        users, items in self.result.shape
        for i in range(users):
            error = 0.0
            for j in range(items):
                error = self.result[i][j] - self.predict[i][j]
                error *= error
            count += error
        return math.sqrt(count / users)

    def pak(self, i, at_k):
        """Return precision@k for the ith row of the result."""
        count = 0.0
        top_predictions = self.predict[i].argsort(
        )[::-1][:at_k]
        result_i = self.result[i]
        for j in top_predictions:
            if result_i[j] != 0.0:
                count += 1
        return count / at_k

    def rak(self, i, at_k):
        """Return recall@k for the ith row of the result."""
        count = 0.0
        top_predictions = self.predict[i].argsort(
        )[::-1][:at_k]
        result_i = self.result[i]
        for j in top_predictions:
            if result_i[j] != 0.0:
                count += 1
        return count / result_i.shape[0]

    def fscore(self, precision, recall):
        """Return fscore."""
        return (2 * precision * recall) /\
            (precision + recall)

    def ndcg(self):
        """To be implemented."""
        pass

    def apak(self, i):
        """Return average precision@k for the ith row of the result."""
        avg_sum = 0.0
        for at_k in range(1, MAX_COMPANION_REC_COUNT * 2):
            p = self.pak(i, at_k=at_k)
            r = self.rak(i, at_k=at_k)
            avg_sum += p * r
        return avg_sum / MAX_COMPANION_REC_COUNT * 2

    def mpak(self):
        """Return mean average precision@k for all rows of the result."""
        _mpak = 0.0
        samples = len(self.test_rows)
        for i in range(samples):
            _apak = self.apak(i)
            _mpak += _apak
        return _mpak / samples

    def loadS3(self):
        """Load user and item matrix from S3."""
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

    def generate_test_data(self):
        """Generate the test data."""
        total_samples = self.theta.shape[0] - 1
        number_of_samples = 0.2 * self.theta.shape[0]
        self.test_rows = random.sample(
            range(total_samples), number_of_samples)
        logger.info(len(self.test_rows))
        logger.info(self.test_rows)
        self.test_data = np.zeros((number_of_samples, K))
        for i, e in enumerate(self.test_rows):
            original_row = self.theta[e]
            self.test_data[i] = original_row
            non_zeros = list(original_row.nonzero()[0])
            len_non_zeros = len(non_zeros)
            if len_non_zeros > 0:
                to_masks = random.randint(1, len_non_zeros)
                vals_to_mask = random.sample(non_zeros, to_masks)
                for j in vals_to_mask:
                    self.test_data[i][j] = 0.0

    def calculate_results(self):
        """Generate the prediction and original result data."""
        self.predict = np.zeros((len(self.test_rows), self.beta.shape[0]))
        self.result = np.zeros((len(self.test_rows), self.beta.shape[0]))
        logger.info(self.result.shape)
        for i, e in enumerate(self.test_rows):
            print(i, e)
            original_row = self.theta[e]
            original_result = Poisson(original_row)
            original_result = original_result.prob(
                self.beta).eval(session=tf.Session())
            original_mean_result = np.zeros(self.beta.shape[0])
            for j in range(self.beta.shape[0]):
                original_mean_result[j] = original_result[j].mean() * 100
            logger.info(self.result[i] + original_mean_result)
            self.result[i] = original_mean_result
            predict_row = self.test_data[i]
            predict_result = Poisson(predict_row)
            predict_result = predict_result.prob(
                self.beta).eval(session=tf.Session())
            predict_mean_result = np.zeros(self.beta.shape[0])
            for j in range(self.beta.shape[0]):
                predict_mean_result[j] = predict_result[j].mean() * 100
            self.predict[i] = predict_mean_result

    def saveS3(self, _mpak):
        """Save the evaluation matric to S3."""
        result_metric = {
            "mpak": _mpak,
            "Test samples": len(self.test_rows),
            "K": K,
            "original size": str(self.theta.shape[0]) + "*" + str(self.beta.shape[0])
        }
        filename = self.generate_uid()
        evaluation_result_filename = os.path.join(
            HPF_SCORING_REGION, HPF_evaluation, filename)
        self.datastore.write_json_file(evaluation_result_filename,
                                       result_metric)

    def execute(self):
        """Run main function."""
        self.loadS3()
        self.generate_test_data()
        self.calculate_results()
        _mpak = self.mpak()
        logger.info("The mean average precision @ K= {}".format(_mpak))
        self.saveS3(_mpak)
