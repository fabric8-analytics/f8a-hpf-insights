FROM centos:7

LABEL maintainer="Sarah Masud <smasud@redhat.com>"

COPY ./src /src
COPY ./requirements.txt /requirements.txt
COPY ./src/config.py.template /src/config.py

ADD ./entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
EXPOSE 6006

RUN yum install -y epel-release &&\
    yum install -y zip gcc git python34-pip python34-requests httpd httpd-devel python34-devel &&\
yum clean all

RUN pip3 install -r /requirements.txt && rm /requirements.txt

ENTRYPOINT ["/bin/entrypoint.sh"]