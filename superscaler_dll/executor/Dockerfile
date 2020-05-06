FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
    g++ \
    git \
    cmake \
    make \
    python

WORKDIR /executor
COPY . /executor
RUN rm -rf build \
    && mkdir build \
    && cd build \
    && cmake .. -DDO_TEST=true \
    && make gtest_parallel