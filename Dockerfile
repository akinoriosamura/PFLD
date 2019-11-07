FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget build-essential gcc zlib1g-dev git default-mysql-client default-libmysqlclient-dev \
    vim \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel

WORKDIR /usr/src/app

RUN pip3 install -U pip
RUN pip install pipenv
