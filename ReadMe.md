Makefile not ready, using compile.sh for now


dummy usage: mpirun -n 2 ./program
rank = (0, 2)
rank = (1, 2)
1.100000, 1.100000, 1.100000, 1.100000,
0.100000, 0.100000, 0.100000, 0.100000,
1.200000, 1.200000, 1.200000, 1.200000,
1.200000, 1.200000, 1.200000, 1.200000,

Distributed Running:
/usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.22 -bind-to none -map-by slot -x CUDA_VISIBLE_DEVICES=2 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib program
