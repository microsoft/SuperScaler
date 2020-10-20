FROM ubuntu:18.04

# Install build environment
RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common

RUN \
    apt-get update && \
    apt-get install -y python3-dev python3-pip &&\
    python3 -m pip install --upgrade pip 

# Create workspace
WORKDIR /Frontend
COPY . /Frontend

# Check code format using flake8 
RUN \
    python3 -m pip install flake8 && \
    python3 -m flake8

# Install library and run pytest
RUN \   
    python3 -m pip install -r requirements.txt && \
    python3 -m pytest -v tests
