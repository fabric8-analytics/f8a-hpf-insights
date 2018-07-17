"""The main program for calling the evaluation class."""

import os
import time
from src.data_store.s3_data_store import S3DataStore
from src.evaluation.evaluation_metric import EvaluationMetric
from src.config import (AWS_S3_ACCESS_KEY_ID,
                        AWS_S3_SECRET_ACCESS_KEY,
                        AWS_S3_BUCKET_NAME,
                        HPF_SCORING_REGION)

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def evaluationS3():
    """Evaluate the matrix stored in S3."""
    t_start = time.time()
    logger.info("HPF evaluation started!")
    s3_object = S3DataStore(src_bucket_name=AWS_S3_BUCKET_NAME,
                            access_key=AWS_S3_ACCESS_KEY_ID,
                            secret_key=AWS_S3_SECRET_ACCESS_KEY)
    em = EvaluationMetric(datastore=s3_object)
    em.execute()
    logger.info("HPF evaluation completed in {} seconds.".format(
        time.time() - t_start))


if __name__ == '__main__':
    evaluationS3()
