FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
    g++ \
    git \
    make \
    python \
    wget

RUN wget -qO- "https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.tar.gz" | \
  tar --strip-components=1 -xz -C /usr/local

WORKDIR /executor
COPY . /executor
RUN rm -rf build \
    && mkdir build \
    && cd build \
    && cmake .. -DDO_TEST=true \
    && make gtest_parallel
