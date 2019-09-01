nvcc -lnccl -c lib_nccl.cu -o libmynccl.o 
mpicc -c lib_mpi.cc -o libmympi.o
gcc -c main.cc -o main.o
mpic++ libmynccl.o libmympi.o main.o -lcudart -lnccl -L/usr/local/cuda/lib64/ -o program
