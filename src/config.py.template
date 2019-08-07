"""The config file."""

import os

# Cloud constants
AWS_S3_ACCESS_KEY_ID = os.environ.get("AWS_S3_ACCESS_KEY_ID", "")
AWS_S3_SECRET_ACCESS_KEY = os.environ.get("AWS_S3_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "")

# Flask endpoint constants
SERVICE_PORT = "6006"
SERVICE_TIMEOUT = "900"
USE_CLOUD_SERVICES = os.environ.get("USE_CLOUD_SERVICES", "False")

# Scoring constants
HPF_SCORING_REGION = os.environ.get("HPF_SCORING_REGION", "maven")
MIN_REC_CONFIDENCE = float(os.environ.get("MIN_REC_CONFIDENCE", "30"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2019-01-03")

# Model filepaths
HPF_output_package_id_dict = os.path.join(MODEL_VERSION,
                                          "trained-model/package_id_dict.json")
HPF_output_manifest_id_dict = os.path.join(MODEL_VERSION,
                                           "trained-model/manifest_id_dict.json")

# Saved Model path
HPF_MODEL_PATH = os.path.join(MODEL_VERSION,
                              "intermediate-model/hpf_model.pkl")

# Swagger path
SWAGGER_YAML_PATH = 'swagger/swagger.yaml'
