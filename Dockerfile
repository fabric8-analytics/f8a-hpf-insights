FROM centos:7

LABEL maintainer="Sarah Masud <smasud@redhat.com>"

RUN yum install -y epel-release &&\
    yum install -y zip gcc git python34-pip python34-requests httpd httpd-devel python34-devel &&\
yum clean all

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && rm /requirements.txt

COPY ./src /src
COPY ./tests/test_data /tests/test_data
COPY ./swagger /src/swagger
COPY ./src/config.py.template /src/config.py

ADD ./entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
EXPOSE 6006

ENTRYPOINT ["/bin/entrypoint.sh"]