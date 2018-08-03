"""The config file."""

import os

# Cloud constants
AWS_S3_ENDPOINT_URL = os.environ.get("AWS_S3_ENDPOINT_URL", "")
AWS_S3_ACCESS_KEY_ID = os.environ.get("AWS_S3_ACCESS_KEY_ID", "")
AWS_S3_SECRET_ACCESS_KEY = os.environ.get("AWS_S3_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "hpf-insights")

# Flask endpoint constants
SERVICE_PORT = "6006"
SERVICE_TIMEOUT = "900"
HPF_SCORING_REGION = os.environ.get("HPF_SCORING_REGION", "maven")

# Training constants
HPF_input_raw_data = "training/manifest.json"
COMPONENT_PREFIX = 'hpf'
BOOTSTRAP_ACTIONS_TEMPLATE = '/deployments/emr_bootstrap.sh'
SOURCE_CODE_ZIPPED_PATH = '/tmp/training.zip'

# Scoring constants
MAX_COMPANION_REC_COUNT = int(os.environ.get("MAX_COMPANION_REC_COUNT", "5"))
UNKNOWN_PACKAGES_THRESHOLD = float(os.environ.get("UNKNOWN_PACKAGES_THRESHOLD", "0.3"))
DEPLOYMENT_PREFIX = os.environ.get("DEPLOYMENT_PREFIX", "dev")

# Model filepaths
HPF_output_package_id_dict = "scoring_test/package_id_dict.json"
HPF_output_manifest_id_dict = "scoring_test/manifest_id_dict.json"
HPF_output_user_matrix = "scoring_test/user_matrix.npz"
HPF_output_item_matrix = "scoring_test/item_matrix.npz"

# Model HyperParameters
a = 0.5
a_c = 0.5
c = 0.5
c_c = 0.5
K = 13
b_c = 0.99
d_c = 0.99
iterations = 101
