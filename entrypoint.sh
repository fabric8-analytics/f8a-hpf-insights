#!/usr/bin/env bash

zip -r /tmp/training.zip /src

gunicorn --pythonpath /src -b 0.0.0.0:$SERVICE_PORT --workers=1 -k sync -t $SERVICE_TIMEOUT flask_endpoint:app