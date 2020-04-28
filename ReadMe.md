1. Build superscaler_dll
Copy superscaler_dll folder to /usr/local/include and run followed commands:

cd /usr/local/include/superscaler_dll
make clean
make

/usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.21 ./test_atomic_operations
/usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.21 ./test
/usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.25 -bind-to none -map-by slot -x CUDA_VISIBLE_DEVICES=2 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib ./test



2. Build Horovod
cd path/to/horovod

virtualenv env
. env/bin/activate

pip install tensorflow==1.14.0
pip install keras==2.2.4
pip install torch==1.1.0 torchvision
pip install pytest mock
pip install h5py future scipy mpi4py pyspark mxnet

cd $HOME
pip uninstall -y horovod
cd -

rm -rf build/ dist/
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_MPI=1 python setup.py install

mpirun -n 2 -H 10.0.0.21,10.0.0.21 python examples/pytorch_mnist.py



3. Issue
(1) nv_peer_memory install
git clone https://github.com/Mellanox/nv_peer_memory.git
cd nv_peer_memory
./build_module.sh
cd /tmp
tar xzf /tmp/nvidia-peer-memory_1.0.orig.tar.gz
cd nvidia-peer-memory-1.0
dpkg-buildpackage -us -uc
cd ..
sudo dpkg -i nvidia-peer-memory_1.0-8_all.deb
sudo dpkg -i nvidia-peer-memory-dkms_1.0-8_all.deb
(2) cuda 10 environment path
export PATH=$PATH:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.0/lib64
source /etc/profile





================================================================================================================================
old version bellow
================================================================================================================================

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
