FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends software-properties-common libgl1-mesa-dev wget libssl-dev

RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get -y install --no-install-recommends python3.9-dev python3.9-distutils python3-pip python3.9-venv
# change default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# clear cache
RUN rm -rf /var/lib/apt/lists/*

# create virtual environment
RUN python -m venv /venv
RUN /venv/bin/pip install -U pip distlib setuptools

# and add aliases
RUN echo 'alias python="/venv/bin/python"' >> ~/.bashrc
RUN echo 'alias pip="/venv/bin/pip"' >> ~/.bashrc
RUN echo 'alias jupyter-lab="/venv/bin/jupyter-lab"' >> ~/.bashrc


WORKDIR /tmp
COPY Makefile .
COPY setup.py .
COPY setup.cfg .
COPY jaxmapp/ jaxmapp/
COPY cython_helper/ cython_helper/
RUN make venv

WORKDIR /workspace

SHELL ["/bin/bash", "-c"]