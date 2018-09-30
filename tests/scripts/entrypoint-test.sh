#!/usr/bin/env bash

locale charmap

export RADONFILESENCODING=UTF-8

echo "*****************************************"
echo "*** Cyclomatic complexity measurement ***"
echo "*****************************************"
radon cc -s -a -i usr .

echo "*****************************************"
echo "*** Maintainability Index measurement ***"
echo "*****************************************"
radon mi -s -i usr .

echo "*****************************************"
echo "*** Unit tests ***"
echo "*****************************************"

mkdir /tmp/hpf # Need to create this dir as shutil.copyfile does not create parents
PYTHONHASHSEED=2 pytest --cov=/src/ --cov-report term-missing -vv /tests/unit_tests/
