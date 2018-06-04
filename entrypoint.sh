#!/usr/bin/env bash

zip -r /tmp/training.zip /src

gunicorn --pythonpath /src -b 0.0.0.0:6006 --workers=2 -k sync -t 900 flask_endpoint:app