#!/usr/bin/env bash

TEST_IMAGE_NAME='hpf-insights-tests'

check_python_version() {
    python3 tools/check_python_version.py 3 6
}

gc() {
    docker rmi -f "$(make get-image-name)"
    docker rmi -f ${TEST_IMAGE_NAME}
}

check_python_version

if [[ "$CI" -eq "0" ]];
then
    make docker-build-test
    docker run ${TEST_IMAGE_NAME}
    docker stop ${TEST_IMAGE_NAME}
    trap gc EXIT SIGINT
else
    # CI instance will be torn down anyway, don't need to waste time on gc
    docker run ${TEST_IMAGE_NAME}
fi
