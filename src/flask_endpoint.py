"""The Endpoint to serve model training and scoring."""

import os
import connexion
import flask
import logging
from flask import request, current_app
from flask_cors import CORS
from rudra.data_store.aws import AmazonS3
from rudra.data_store.local_data_store import LocalDataStore
from src.scoring.hpf_scoring import HPFScoring
from src.utils import convert_string2bool_env
from src.config import (AWS_S3_ACCESS_KEY_ID,
                        AWS_S3_SECRET_ACCESS_KEY,
                        HPF_SCORING_REGION,
                        AWS_S3_BUCKET_NAME,
                        USE_CLOUD_SERVICES,
                        SWAGGER_YAML_PATH)
from raven.contrib.flask import Sentry

# Turn off the annoying boto logs unless some error occurs
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('s3transfer').setLevel(logging.ERROR)
logging.getLogger('boto3').setLevel(logging.ERROR)


def setup_logging(flask_app):  # pragma: no cover
    """Perform the setup of logging (file, log level) for this application."""
    if not flask_app.debug:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
        log_level = os.environ.get(
            'FLASK_LOGGING_LEVEL', logging.getLevelName(logging.INFO))
        handler.setLevel(log_level)

        flask_app.logger.addHandler(handler)
        flask_app.config['LOGGER_HANDLER_POLICY'] = 'never'
        flask_app.logger.setLevel(logging.DEBUG)


app = connexion.FlaskApp(__name__)
setup_logging(app.app)
CORS(app.app)
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
sentry = Sentry(app, dsn=SENTRY_DSN, logging=True, level=logging.ERROR)
app.logger.info('App initialized, ready to roll...')

global scoring_status
global scoring_object
global s3_object

if HPF_SCORING_REGION != "":
    if convert_string2bool_env(USE_CLOUD_SERVICES):
        s3_object = AmazonS3(bucket_name=AWS_S3_BUCKET_NAME,
                             aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY)
        s3_object.connect()
        app.scoring_object = HPFScoring(datastore=s3_object)
    else:
        app.scoring_object = HPFScoring(LocalDataStore("tests/test_data"))
    app.scoring_status = True
else:
    app.scoring_status = False
    current_app.logger.warning("Have not loaded a model for scoring!")


def heart_beat():
    """Handle the / REST API call."""
    return flask.jsonify({"status": "ok"})


def liveness():
    """Define the linveness probe."""
    return flask.jsonify({"status": "alive"})


def readiness():
    """Define the readiness probe."""
    if not app.scoring_status:
        return flask.jsonify({"error": "Could not load model from S3"}), 500
    return flask.jsonify({"status": "ready"})


def hpf_scoring():
    """Endpoint to serve recommendations."""
    response_json = []
    if app.scoring_status:
        input_json = request.get_json()
        for input_stack in input_json:
            output_json = dict()
            if input_stack["ecosystem"] != HPF_SCORING_REGION:  # pragma: no cover
                output_json = {"Error": "Input ecosystem does not match"}
            else:
                companion_recommendation, package_to_topic_dict, \
                    missing_packages = app.scoring_object.predict(
                        input_stack['package_list'])
                current_app.logger.debug(
                    'The companion recommendation are {}'.format(companion_recommendation))
                output_json = {
                    "alternate_packages": {},
                    "missing_packages": missing_packages,
                    "companion_packages": companion_recommendation,
                    "ecosystem": input_stack["ecosystem"],
                    "package_to_topic_dict": package_to_topic_dict,
                }
            response_json.append(output_json)
        return flask.jsonify(response_json)
    else:  # pragma: no cover
        current_app.logger.error(
            'No scoring region provided. HPF_SCORING_REGION is {}'.format(HPF_SCORING_REGION))
        response_json.append(
            {"Error": "No scoring region provided"})
        return flask.jsonify(response_json)


def hpf_model_details():
    """Endpoint to return model size details."""
    if app.scoring_status:
        return flask.jsonify({"Model Details": app.scoring_object.model_details()})
    else:  # pragma: no cover
        return flask.jsonify({"Error": "No scoring region provided"})


def log_it(func):
    """Func decorator for logging."""
    def inner1(*args, **kwargs):
        app.logger.info("Executed {}".format(func.__name__))
        return func(*args, **kwargs)
    return inner1


app.add_api(SWAGGER_YAML_PATH)

if __name__ == "__main__":
    app.run(debug=True, port=6006)
