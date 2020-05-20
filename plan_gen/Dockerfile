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
WORKDIR /Plan_Gen
COPY . /Plan_Gen

# Check code format using flake8 
RUN \
    python3 -m pip install flake8 && \
    python3 -m flake8

# Install library and run pytest
RUN \
    python3 -m pip install -r requirements.txt && \
    python3 -m pytest -v
