# f8a-hpf-insights
**(fabric8-analytics-hpf-insights)**

HPF Matrix Factorizations for companion recommendation.

*HPF- Hierarchical Poisson Factorization*

## Supported ecosystems:
* Maven - Last trained at: 2018-08-08 11:30 IST(UTC +5:30)

## Build upon:
* https://github.com/arindamsarkar93/hcpf

## To run locally via docker-compose:

* Setup Minio and start Minio server so that `hpf-insights` is loaded as a folder inside it upon running. To use AWS S3 instead of Minio add your AWS S3 credentials in the next step instead of Miino credentials.
* Create a `.env` file and add credentials to it.
* In the `.env` set the `AWS_S3_ENDPOINT_URL` to `<blank>` for using AWS S3 and to `http://ip:port` for using Minio.
* `source .env`
* `docker-compose build`
* `docker-compose up`
* `curl  http://0.0.0.0:6006/` should return `status: ok`


## To run on dev-cluster:

* `cp secret.yaml.template secret.yaml`
* Add your AWS S3 credentials to `secret.yaml`
* `oc login`
* `oc new-project hpf-insights`
* `oc create -f secret.yaml`
* `oc process -f openshift/template.yaml -o yaml|oc create -f -` If you want to update the template.yaml and redeploy it, then do `oc process -f openshift/template.yaml -o yaml|oc apply -f -` Use apply instead of create for subsequent re-deployments.
* Go your Openshift console and expose the route
* `curl <route_URL>` should return `status:ok`


## To run load testing for recommendation API:

* `pip install locustio==0.8.1`
* Bring up the service.
* `locust -f perf_tests/locust_tests.py --host=<URL of the service>`

### Footnotes:

#### Coding standards:

- You can use scripts `run-linter.sh` and `check-docstyle.sh` to check if the code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [PEP 257](https://www.python.org/dev/peps/pep-0257/) coding standards. These scripts can be run w/o any arguments:

```
./run-linter.sh
./check-docstyle.sh
```

The first script checks the indentation, line lengths, variable names, whitespace around operators etc. The second
script checks all documentation strings - its presense and format. Please fix any warnings and errors reported by these
scripts.

#### Code complexity measurement:

The scripts `measure-cyclomatic-complexity.sh` and `measure-maintainability-index.sh` are used to measure code complexity. These scripts can be run w/o any arguments:

```
./measure-cyclomatic-complexity.sh
./measure-maintainability-index.sh
```

The first script measures cyclomatic complexity of all Python sources found in the repository. Please see [this table](https://radon.readthedocs.io/en/latest/commandline.html#the-cc-command) for further explanation how to comprehend the results.

The second script measures maintainability index of all Python sources found in the repository. Please see [the following link](https://radon.readthedocs.io/en/latest/commandline.html#the-mi-command) with explanation of this measurement.

#### Dead code detection

The script `detect-dead-code.sh` can be used to detect dead code in the repository. This script can be run w/o any arguments:

```
./detect-dead-code.sh
```

Please note that due to Python's dynamic nature, static code analyzers are likely to miss some dead code. Also, code that is only called implicitly may be reported as unused.

Because of this potential problems, only code detected with more than 90% of confidence is reported.

#### Common issues detection

The script `detect-common-errors.sh` can be used to detect common errors in the repository. This script can be run w/o any arguments:

```
./detect-common-errors.sh
```

Please note that only semantical problems are reported.

#### Check for scripts written in BASH

The script named `check-bashscripts.sh` can be used to check all BASH scripts (in fact: all files with the `.sh` extension) for various possible issues, incompatibilies, and caveats. This script can be run w/o any arguments:

```
./check-bashscripts.sh
```

Please see [the following link](https://github.com/koalaman/shellcheck) for further explanation, how the ShellCheck works and which issues can be detected.

### Additional links:
* [Pushing Image to Docker Hub](https://ropenscilabs.github.io/r-docker-tutorial/04-Dockerhub.html)
* [PAPER: Scalable Recommendation with Poisson Factorization](https://arxiv.org/abs/1311.1704)
* [PAPER: Hierarchical Compound Poisson Factorization](https://arxiv.org/abs/1604.03853)
