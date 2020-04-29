FROM ubuntu:16.04

# Install build environment
RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN \
    apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3-pip &&\
    python3.6 -m pip install --upgrade pip && \
    # Change python 3.6 to default
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2   

# Create workspace
WORKDIR /AISimulator
COPY . /AISimulator
# Install AISimulator
RUN \
    python3 -m pip install flake8 && \
    python3 -m flake8 && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pytest -v
