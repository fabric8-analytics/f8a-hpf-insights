#!/usr/bin/env bash

zip -r /tmp/training.zip /src

#NOTE: Set worker to 1, as single 1.5 mb pod can support only 1 900 mb model
gunicorn --pythonpath /src -b 0.0.0.0:6006 --workers=1 -k sync -t 900 flask_endpoint:app