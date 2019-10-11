# run rdma (cpu & gpu) test
scp -r /home/shamao/SuperScaler shamao@MSRAGPUM24:/home/shamao/
/usr/bin/mpirun -n 2 -H 10.0.0.23,10.0.0.24 -bind-to none -map-by slot -x CUDA_VISIBLE_DEVICES=2 -x NCCL_DEBUG=info --x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^lo,docker0,eth0,eth2,br-a4f781fec04e,br-983805864fb9 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0,eth0,eth2,br-a4f781fec04e,br-983805864fb9 ./final_test
