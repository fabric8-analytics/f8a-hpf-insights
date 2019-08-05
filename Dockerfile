FROM centos:7

LABEL maintainer="Lucky Suman <lsuman@redhat.com>"

RUN yum install -y epel-release &&\
    yum install -y zip gcc-c++ git python36-pip python36-requests httpd httpd-devel python36-devel &&\
    yum clean all

COPY ./requirements.txt /requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install git+https://github.com/fabric8-analytics/fabric8-analytics-rudra#egg=rudra
RUN pip3 install -r /requirements.txt && rm /requirements.txt

COPY ./src /src
COPY ./tools /tools
COPY ./tests/test_data /tests/test_data
COPY ./src/config.py.template /src/config.py

ADD ./entrypoint.sh /bin/entrypoint.sh
RUN chmod +x /bin/entrypoint.sh
RUN pip3 install Cython==0.29.1 && pip3 install hpfrec==0.2.2.9
EXPOSE 6006

ENTRYPOINT ["/bin/entrypoint.sh"]
