"""The Endpoint to serve model training and scoring."""

import os
import flask
import logging
from flask import Flask, request, current_app
from flask_cors import CORS
from src.data_store.s3_data_store import S3DataStore
from src.scoring.hpf_scoring import HPFScoring
from src.config import (AWS_S3_ACCESS_KEY_ID,
                        AWS_S3_SECRET_ACCESS_KEY,
                        AWS_S3_BUCKET_NAME,
                        HPF_SCORING_REGION)


def setup_logging(flask_app):
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


app = Flask(__name__)
setup_logging(app)
CORS(app)

global scoring_status
global scoring_object
global s3_object


if HPF_SCORING_REGION != "":
    s3_object = S3DataStore(src_bucket_name=AWS_S3_BUCKET_NAME,
                            access_key=AWS_S3_ACCESS_KEY_ID,
                            secret_key=AWS_S3_SECRET_ACCESS_KEY)
    app.scoring_object = HPFScoring(datastore=s3_object)
    app.scoring_status = True
else:
    app.scoring_status = False


def list_routes():
    """Return a list of routes for this app."""
    return [str(rule) for rule in app.url_map.iter_rules()]


@app.route('/')
def heart_beat():
    """Handle the / REST API call."""
    return flask.jsonify({"status": "ok", "routes": list_routes()})


@app.route('/api/v1/liveness', methods=['GET'])
def liveness():
    """Define the linveness probe."""
    return flask.jsonify({"status": "alive"})


@app.route('/api/v1/readiness', methods=['GET'])
def readiness():
    """Define the readiness probe."""
    return flask.jsonify({"status": "ready"})


@app.route('/api/v1/companion_recommendation', methods=['POST'])
def hpf_scoring():
    """Endpoint to serve recommendations."""
    response_json = []
    if app.scoring_status:
        input_json = request.get_json()
        for input_stack in input_json:
            output_json = dict()
            if input_stack["ecosystem"] != HPF_SCORING_REGION:
                output_json = {"Error": "Ecosystems don't match. \
                    GIVEN:{} EXPECTED:{}".format(input_stack["ecosystem"],
                                                 HPF_SCORING_REGION)}
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
    else:
        current_app.logger.error(
            'No scoring region provided. HPF_SCORING_REGION is {}'.format(HPF_SCORING_REGION))
        response_json.append(
            {"Error": "No scoring region provided"})
    return flask.jsonify(response_json)


@app.route('/api/v1/model_details', methods=['GET'])
def hpf_model_details():
    """Endpoint to return model size details."""
    if app.scoring_status:
        return flask.jsonify({"Model Details": app.scoring_object.model_details()})
    else:
        return flask.jsonify({"Error": "No scoring region provided"})


if __name__ == "__main__":
    app.run()
