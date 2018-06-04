# poc-maven-stack-analysis
HPF Matrix Factorizations for Maven companion recommendation

## Build upon
* https://github.com/arindamsarkar93/hcpf

## TO run locally via Docker Compose

* Setup Minio and start Minio server so that `hpf-insights` is loaded as a folder inside it upon running. To use AWS S3 instead of Minio add your AWS S3 credentials in the next step instead of Miino credentials.
* Create a `.env` file and add credentials to it.
* In the `.env` set the `AWS_S3_ENDPOINT_URL` to `<blank>` for using AWS S3 and to `http://ip:port` for using Minio.
* `source .env`
* `docker-compose build`
* `docker-compose up`
* `curl  http://0.0.0.0:6006/` should return `status: ok`


## To run on DevCluster

* `cp secret.yaml.template secret.yaml`
* Add your AWS S3 credentials to `secret.yaml`
* `oc login`
* `oc new-project hpf-insights`
* `oc create -f secret.yaml`
* `oc process -f openshift/template.yaml -o yaml|oc create -f -` If you want to update the template.yaml and redeploy it, then do `oc process -f openshift/template.yaml -o yaml|oc apply -f -` Use apply instead of create for subsequent re-deployments.
* Go your Openshift console and expose the route
* `curl <route_URL>` should return `status:ok`