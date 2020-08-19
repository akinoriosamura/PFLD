FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    wget build-essential gcc zlib1g-dev git \
    vim \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopencv-dev \
    libboost-all-dev \
    libhdf5-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libopenblas-dev \
    libcaffe-cpu-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel

WORKDIR /usr/src/app

RUN pip3 install -U pip
RUN pip install pipenv
