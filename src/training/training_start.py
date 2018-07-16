"""The offline training script."""

import os
import time
from src.data_store.local_data_store import LocalDataStore
from src.data_store.s3_data_store import S3DataStore
from src.training.data_preprocessing import DataPreprocessing
from src.training.hyper_parameter_tuning import HyperParameterTuning
from src.training.generate_matrix import GenerateMatrix
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
    t_start = time.time()
    logger.info("HPF training started!")
    try:
        if not os.path.exists("/tmp/hpf"):
            os.mkdir("/tmp/hpf")
            logger.info("HPF directory created.")
        else:
            logger.info("HPF directory exits using it.")
    except Exception as e:
        logger.info("Could not create temp HPF directory.")
        exit(1)
    s3_object = S3DataStore(src_bucket_name=AWS_S3_BUCKET_NAME,
                            access_key=AWS_S3_ACCESS_KEY_ID,
                            secret_key=AWS_S3_SECRET_ACCESS_KEY)
    dp = DataPreprocessing(datastore=s3_object)
    t0 = time.time()
    logger.info(
        "Generation of rating matix, package_id_dict and manifest_id_dict")
    dp.execute()
    logger.info("Preprocessing Ended in {} seconds".format(time.time() - t0))
    t0 = time.time()
    logger.info("HyperParameter tuning starts now.")
    hp = HyperParameterTuning()
    hp.execute()
    logger.info(
        "HyperParameter Tuning Ended in {} seconds".format(time.time() - t0))
    t0 = time.time()
    logger.info("Started generating matrices.")
    gm = GenerateMatrix(datastore=s3_object)
    gm.execute()
    logger.info(
        "Generation of matrices ended in {} seconds".format(time.time() - t0))
    t0 = time.time()
    logger.info("HPF training completed in {} seconds.".format(
        time.time() - t_start))


if __name__ == '__main__':
    trainingS3()
