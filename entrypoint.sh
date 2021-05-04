#!/usr/bin/env bash

mkdir -p /tmp/hpf/

gunicorn -b 0.0.0.0:6006 --workers=1 -k sync -t 900 src.flask_endpoint:app --log-level $FLASK_LOGGING_LEVEL
