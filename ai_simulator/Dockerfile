FROM ubuntu:16.04

# Install build environment
RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install python3-dev python3-pip && \
    python3 -m pip install --upgrade pip
# Create workspace
WORKDIR /AISimulator
COPY . /AISimulator
# Install AISimulator
RUN \
    python3 -m pip install -r requirements.txt && \
    python3 -m pytest -v
 



