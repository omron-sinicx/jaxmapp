FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends software-properties-common libgl1-mesa-dev wget libssl-dev

RUN apt-get -y install --no-install-recommends python3.8-dev python3.8-distutils python3-pip python3.8-venv
# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# clear cache
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip distlib setuptools wheel


WORKDIR /tmp
COPY README.md .
COPY setup.py .
COPY pyproject.toml .
COPY src/ src/
RUN pip3 install -e .[dev]

WORKDIR /workspace

SHELL ["/bin/bash", "-c"]
