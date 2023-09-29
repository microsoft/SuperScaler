FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

RUN pip install -U pip setuptools 
RUN pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch/
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt

COPY external/apex /workspace/apex
WORKDIR /workspace/apex
RUN pip install -r requirements.txt
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./

COPY external/alpa /workspace/alpa
WORKDIR /workspace/alpa
RUN pip uninstall -y cupy-cuda115
RUN pip install -r requirements.txt

RUN pip install jaxlib==0.3.5+cuda113.cudnn820 -f https://github.com/alpa-projects/alpa/releases/download/v0.1.5/jaxlib-0.3.5%2Bcuda113.cudnn820-cp38-none-manylinux2010_x86_64.whl

RUN apt-get update && apt-get -y install pssh
RUN apt-get -y install coinor-cbc

WORKDIR /workspace