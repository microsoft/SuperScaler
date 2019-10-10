
NVCC=nvcc 

CUDA ?= /usr/local/cuda-10

MPI_LIB := -I/usr/mpi/gcc/openmpi-2.0.2a1/include/ -L/usr/mpi/gcc/openmpi-2.0.2a1/ -lmpi -lnccl -Xcompiler "-fPIC"
CUDA_LIB := -L./ -L$(CUDA)/lib64 -L/var/drivers/nvidia/367.55/lib64  -L/usr/lib
CUDA_INC += -I./ -I$(CUDA)/include -I/var/drivers/nvidia/367.55/include -I/usr/include

CPP11 := -std=c++11 
CPPFLAGS := -std=c++11 $(CUDA_INC)
LDFLAGS  := $(CUDA_LIB) 
CFLAGS   += -O0
CXXFLAGS += -O0
CFLAGS   += -g
CXXFLAGS += -g
LIBS     := -libverbs -lrdmacm -lcudart -lcuda -lpthread -ldl
#LIBS     := -libverbs -lrdmacm -lcudart -lcuda -lpthread -ldl -lgdrapi

SRCS := gdr_srv.cc gdr_cli.cc
EXES := $(SRCS:.cc=)

all: super_scaler_all_reduce lib_super_scaler_all_reduce final_test

super_scaler_all_reduce: super_scaler_all_reduce.cu config_parse/parse.cc blas/blas.cu
	$(NVCC) $(MPI_LIB) $(CPP11) $^ -c

lib_super_scaler_all_reduce: super_scaler_all_reduce.o
	$(NVCC) $(CPP11) -shared $^ -o $@.so

final_test: test_lib.cc rdma/rdma_common.cc rdma/rdma_device.cc rdma/rdma_device_manager.cc rdma/rdma_thread.cc rdma/rdma.cc config_parse/parse.cc blas/blas.cu primitive_operation/comm_primitive.cc
	$(NVCC) $(CXXFLAGS) $(CPPFLAGS) $(MPI_LIB) $(LDFLAGS) -o $@ $^ $(LIBS) -L. -l_super_scaler_all_reduce

clean:
	rm -f *.o $(EXES) lib*.{a,so}* *~ core.*

.PHONY: driver clean all lib exes lib_install install
