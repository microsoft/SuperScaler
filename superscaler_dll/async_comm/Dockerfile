FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
    g++ \
    git \
    cmake \
    make \
    python

WORKDIR /async_common_lib
COPY . /async_common_lib
RUN rm -rf build \
    && mkdir build \
    && cd build \
    && cmake .. -DDO_TEST=true \
    && make gtest_parallel