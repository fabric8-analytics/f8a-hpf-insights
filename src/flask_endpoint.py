"""The Endpoint to serve model training and scoring."""

import os
import flask
from flask import Flask, request
from flask_cors import CORS
from src.data_store.s3_data_store import S3DataStore
from src.scoring.hpf_scoring import HPFScoring
from src.config import (AWS_S3_ACCESS_KEY_ID,
                        AWS_S3_SECRET_ACCESS_KEY,
                        AWS_S3_BUCKET_NAME,
                        HPF_SCORING_REGION,
                        SCORING_THRESHOLD)


app = Flask(__name__)
CORS(app)

global scoring_status
global scoring_object
global s3_object


if HPF_SCORING_REGION != "":
    s3_object = S3DataStore(src_bucket_name=AWS_S3_BUCKET_NAME,
                            access_key=AWS_S3_ACCESS_KEY_ID,
                            secret_key=AWS_S3_SECRET_ACCESS_KEY)
    app.scoring_object = HPFScoring(datastore=s3_object,
                                    scoring_threshold=SCORING_THRESHOLD,
                                    scoring_region=HPF_SCORING_REGION)
    app.scoring_status = True
else:
    app.scoring_status = False


@app.route('/')
def heart_beat():
    """Handle the / REST API call."""
    return flask.jsonify({"status": "ok"})


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
            if input_stack["ecosystem"] != HPF_SCORING_REGION:
                response_json.append(
                    {"Error": "Ecosystems don't match. \
                    GIVEN:{} EXPECTED:{}".format(input_stack["ecosystem"],
                                                 HPF_SCORING_REGION)})
            else:
                companion_recommendation, package_to_topic_dict, missing_packages =\
                    app.scoring_object.predict(
                        input_stack['package_list'])
                response_json.append({
                    "missing_packages": missing_packages,
                    "companion_packages": companion_recommendation,
                    "ecosystem": input_stack["ecosystem"],
                    "package_to_topic_dict": package_to_topic_dict
                })
    else:
        response_json.append({"Error": "No scoring region provided"})
    return flask.jsonify(response_json)


if __name__ == "__main__":
    app.run()
