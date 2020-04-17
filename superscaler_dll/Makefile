NVCC=nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -DWARNING_AS_ERROR

CUDA ?= /usr/local/cuda-10

MPI_LIB := -I/usr/mpi/gcc/openmpi-2.0.2a1/include/ -L/usr/mpi/gcc/openmpi-2.0.2a1/ -lmpi -lnccl -Xcompiler "-fPIC"
CUDA_LIB := -L./ -L$(CUDA)/lib64 -L/var/drivers/nvidia/367.55/lib64  -L/usr/lib
CUDA_INC += -I./ -I$(CUDA)/include -I/var/drivers/nvidia/367.55/include -I/usr/include

CPP11 := -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0
CPPFLAGS := -std=c++14 $(CUDA_INC)
LDFLAGS  := $(CUDA_LIB)

ifeq ($(BUILD),debug)   
CFLAGS += -O0 -g
CXXFLAGS += -O0 -g
else
CFLAGS += -O3 -DNDEBUG
CXXFLAGS += -O3 -DNDEBUG
endif

LIBS     := -libverbs -lrdmacm -lcudart -lcuda -lcublas -lpthread -ldl

SRCS := gdr_srv.cc gdr_cli.cc
EXES := $(SRCS:.cc=)

all: super_scaler lib_super_scaler.a test_atomic_operations test

debug:
	make "BUILD=debug"

super_scaler: super_scaler.cc plan_parse/plan_parse.cc atomic_operations/comm_primitive.cc config_parse/parse.cc blas/blas.cu nccl/nccl.cc \
			  rdma/rdma_common.cc rdma/rdma_device.cc rdma/rdma_device_manager.cc rdma/rdma_thread.cc rdma/rdma.cc \
			  cuda_ipc/shared_pipe.cpp cuda_ipc/shared_block.cpp cuda_ipc/cuda_ipc_comm_primitive.cpp
	$(NVCC) $(MPI_LIB) $(CPP11) $^ -c

lib_super_scaler.a: super_scaler.o plan_parse.o comm_primitive.o parse.o blas.o nccl.o \
					rdma_common.o rdma_device.o rdma_device_manager.o rdma_thread.o rdma.o \
					shared_pipe.o shared_block.o cuda_ipc_comm_primitive.o
	ar crv $@ $^

test_atomic_operations: atomic_operations/test.cc
	$(NVCC) $(CXXFLAGS) $(CPPFLAGS) $(MPI_LIB) $(LDFLAGS) -o $@ $^ $(LIBS) -L. -l_super_scaler

test: test.cc
	$(NVCC) $(CXXFLAGS) $(CPPFLAGS) $(MPI_LIB) $(LDFLAGS) -o $@ $^ $(LIBS) -L. -l_super_scaler

clean:
	rm -f *.o $(EXES) lib*.{a,so}* *~ core.* *.so *.a test_atomic_operations test

.PHONY: driver clean all lib exes lib_install install
