FROM ubuntu:18.04

# Install build environment
RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common

# Install python3.6 and pip3
RUN \
    apt-get update -y && \
    apt-get install -y python3-dev && \
    apt-get install -y python3-pip && \
    python3 -m pip install --upgrade pip 

# Create workspace
WORKDIR /SuperScaler
COPY . /SuperScaler

# Check code format using flake8 
RUN \
    python3 -m pip install flake8 && \
    python3 -m flake8 tests

# Install tensorflow which currently used by plan_gen
RUN \
    python3 -m pip install tensorflow==1.15

# Install superscaler as a package
RUN \
    python3 -m pip install --upgrade setuptools==49.6.0 && \
    python3 setup.py install

# Run pytest
RUN \
    python3 -m pytest -v tests