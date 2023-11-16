# Written by Seongmoon Jeong - 2022.04.29

FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Use kakao mirror
RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

# Temporarily added for issue.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    ca-certificates git \
    cmake protobuf-compiler ninja-build \
    python3.7-dev python3-pip python3-pil python3-opencv python3-lxml
RUN ln -sv /usr/bin/python3.7 /usr/bin/python

# Use kakao mirror on pypi
RUN python -m pip install -U pip 
RUN python -m pip config --user set global.index "http://mirror.kakao.com/pypi/pypi"
RUN python -m pip config --user set global.index-url "http://mirror.kakao.com/pypi/simple"
RUN python -m pip config --user set global.trusted-host "mirror.kakao.com"

# Install pytorch & detectron2
RUN python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f \
    "https://download.pytorch.org/whl/torch_stable.html"
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="8.0"
RUN python -m pip install detectron2==0.5 -f \
    "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"

# Install tensorflow & object_dection API
RUN python -m pip install tensorflow-gpu==2.4.0
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
RUN ln -s libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10
WORKDIR /root
RUN git clone https://github.com/tensorflow/models.git
RUN (cd /root/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /root/models/research
RUN cp object_detection/packages/tf2/setup.py .
RUN python -m pip install .

# Install others
RUN apt-get install -y \
    tree curl wget nano vim htop screen default-jdk
RUN python -m pip install \
    parmap scikit-image seaborn ray[default] compressai albumentations fiftyone \
    setuptools==58.2.0
RUN pip install -U pip
ENV TZ=Asia/Seoul

WORKDIR /surrogate
