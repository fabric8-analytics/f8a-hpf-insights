#!/usr/bin/env bash

zip -r /tmp/training.zip /src

#NOTE: Set worker to 1, as single 1.5 mb pod can support only 1 900 mb model
gunicorn --preload True --pythonpath /src -b 0.0.0.0:$SERVICE_PORT --workers=1 -k sync -t $SERVICE_TIMEOUT flask_endpoint:app