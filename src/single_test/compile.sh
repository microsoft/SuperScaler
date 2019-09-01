nvcc -I/usr/mpi/gcc/openmpi-2.0.2a1/include/ -L/usr/mpi/gcc/openmpi-2.0.2a1/ -lmpi -lnccl run_nccl.cu -o program
