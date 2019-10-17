# `FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04` でベースにする CUDA 環境を選ぶことができる。
# 利用可能な CUDA 環境の一覧は https://hub.docker.com/r/nvidia/cuda/ にある。
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# 自動アップグレード
RUN apt-get update -y
RUN apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" dist-upgrade

# 使いたいソフトウェアを入れる
RUN apt-get install -y --no-install-recommends \
    sudo ssh \
    build-essential \
    zsh screen cmake unzip git curl wget vim tree htop \
    python-dev python-pip python-setuptools \
    python3-dev python3-pip python3-setuptools \
    graphviz

# キャッシュを消してイメージを小さくする
RUN apt-get clean -y
RUN apt-get autoremove -y
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get autoremove -y
RUN apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# 以後 ubuntu ユーザの設定
USER ubuntu
WORKDIR /home/ubuntu
ENV HOME /home/ubuntu

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR $HOME/