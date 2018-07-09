"""The offline training script."""

import os
import time
from src.data_store.local_data_store import LocalDataStore
from src.data_store.s3_data_store import S3DataStore
from src.training.data_preprocessing import DataPreprocessing
from src.config import (AWS_S3_ACCESS_KEY_ID,
                        AWS_S3_SECRET_ACCESS_KEY,
                        AWS_S3_BUCKET_NAME,
                        HPF_SCORING_REGION)
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def trainingS3():
    """Call the preprocess module. with S3 as datastore."""
    s3_object = S3DataStore(src_bucket_name=AWS_S3_BUCKET_NAME,
                            access_key=AWS_S3_ACCESS_KEY_ID,
                            secret_key=AWS_S3_SECRET_ACCESS_KEY)
    dp = DataPreprocessing(datastore=s3_object,
                           scoring_region=HPF_SCORING_REGION)
    t0 = time.time()
    logger.info(
        "Generation of rating matix, package_id_dict and manifest_id_dict")
    dp.generate_original_rating_matrix()
    logger.info("Preprocessing Ended in {} seconds".format(time.time() - t0))


if __name__ == '__main__':
    trainingS3()
