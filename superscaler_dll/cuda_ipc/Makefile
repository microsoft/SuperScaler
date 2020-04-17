CUDA_PATH   := /usr/local/cuda
NVCC        := $(CUDA_PATH)/bin/nvcc -ccbin g++ -std=c++14 -O3 -g --compiler-options -Wall --compiler-options -Werror -DWARNING_AS_ERROR
LDFLAGS     := -lcuda -lcublas -lnccl -lpthread
INCLUDE     := -I $(CUDA_PATH)/include
SRC         := shared_pipe.cpp shared_block.cpp cuda_ipc_comm_primitive.cpp

all: cuda_ipc_test.exe

cuda_ipc_test.exe: cuda_ipc_test.cpp $(SRC)
	$(NVCC) -DTEST $(LDFLAGS) $(INCLUDE) $(GENCODE) -o $@ $^

test: cuda_ipc_test.exe
	./cuda_ipc_test.exe

clean:
	rm -f *.exe
