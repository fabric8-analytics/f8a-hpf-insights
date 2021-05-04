#!/usr/bin/env bash

zip -r /tmp/training.zip /src
mkdir -p /tmp/hpf/

gunicorn --pythonpath /src -b 0.0.0.0:6006 --workers=1 -k sync -t 900 flask_endpoint:app
