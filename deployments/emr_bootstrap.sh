#!/usr/bin/env bash

set -ex

sudo yum -y install -v python36 python36-pip

sudo python3.6 -m pip install \
    flask \
    flask-cors \
    edward==1.3.5 \
    tensorflow==1.4.1 \
    numpy \
    boto3  \
    botocore \
    gunicorn \
    scipy

export PYTHONPATH='/home/hadoop/'
