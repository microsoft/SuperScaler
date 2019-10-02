#include <stdio.h>
#include <iostream>
#include "super_scaler.h"
#include <chrono>

#include "rdma.h"

void callBackFunction()
{
    std::cout << "Call Back Success!" << std::endl;
}

void test_host(int myRank, int nRanks, int localRank, int size)
{
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = i;
    }

    //each process use 1 GPU
    int nDev = 1;

    //initializing GPU memery based on localRank
    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(localRank * nDev + i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));

        CUDACHECK(cudaStreamCreate(s + i));
    }

    //generating NCCL unique ID at one process and broadcasting it to all
    ncclUniqueId id;
    if (myRank == 0)
    {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL, group API is required around ncclCommInitRank as it is called across multiple GPUs in each thread/process
    ncclComm_t comms[nDev];
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaSetDevice(localRank * nDev + i));
        NCCLCHECK(ncclCommInitRank(comms + i, nRanks * nDev, id, myRank * nDev + i));
    }
    NCCLCHECK(ncclGroupEnd());

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
        super_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank,
                                     sendbuff, recvbuff, comms, s);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / 10;
	std::cout << "test_host_nccl, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	 

    //freeing device memory
    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for (int i = 0; i < nDev; i++)
    {
        ncclCommDestroy(comms[i]);
    }


    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        if(gradients[i] != i )
        {
            std::cout <<  "test_host fail " <<gradients[i] << " " << (nRanks)*(nRanks-1)/2 << "\n" ;
            break;
        }
    }
}

void test_rdma(int myRank, int nRanks, int localRank, int size)
{
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = i;
    }

    float **sendbuff = (float **)malloc(1 * sizeof(float *));
    float **recvbuff = (float **)malloc(1 * sizeof(float *));

    CUDACHECK(cudaSetDevice(localRank * 1 + 0));
    CUDACHECK(cudaMalloc(sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[0], 1, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff[0], gradients, size * sizeof(float), cudaMemcpyHostToDevice));

    set_cfg_RDMA(global_cfg, myRank, nRanks, localRank, gradients, sendbuff[0], recvbuff[0], size);
/*
    std::cout << "Before all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << gradients[i] << " ";
    }
    std::cout << std::endl;
*/
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
    RDMA_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time)/10;
	std::cout << "test_rdma, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	 

    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        if(gradients[i] != i )
        {
            std::cout <<  "test_rdma fail " <<gradients[i] << " " << i << "\n" ;
            break;
        }
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
    RDMA_scaler_all_reduce_device(sendbuff[0], size, myRank, nRanks, localRank);
    elapsed_seconds = (std::chrono::system_clock::now() - start_time)/10;
	std::cout << "test_rdma, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	 

}

void test_mpi_host(int myRank, int nRanks, int localRank, int size)
{
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = i;
    }
/*
    std::cout << "Before all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << gradients[i] << " ";
    }
    std::cout << std::endl;
*/
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
    {
        MPI_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank);
    }
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / 10;
	std::cout << "test_mpi_host, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	
    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        if(gradients[i] != i )
        {
            std::cout <<  "test_host fail " <<gradients[i] << " " << (nRanks)*(nRanks-1)/2 << "\n" ;
            break;
        }
    }
}

void test_USR_host(int myRank, int nRanks, int localRank, int size)
{
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = i;
    }
/*
    std::cout << "Before all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << gradients[i] << " ";
    }
    std::cout << std::endl;
*/
    auto plan = global_cfg.cfg_table["allreduce.classifier.6.bias"];
    void* output_ptr = malloc(size*sizeof(float));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
        MPI_usr_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank, plan, output_ptr);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / 10;
	std::cout << "test_mpi_host, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	delete[] output_ptr;
    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        if(gradients[i] != i*2 )
        {
            std::cout <<  "test_host fail " <<gradients[i] << " " << i*2 << "\n" ;
            break;
        }
    }
}

void test_device(int myRank, int nRanks, int localRank, int size)
{
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = i ;
    }
/*
    std::cout << "Before all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        std::cout << gradients[i] << " ";
    }
    std::cout << std::endl;
*/
    //initializing GPU memery based on localRank
    float **sendbuff = (float **)malloc(1 * sizeof(float *));

    CUDACHECK(cudaSetDevice(localRank * 1 + 0));
    CUDACHECK(cudaMalloc(sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[0], 1, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff[0], gradients, size * sizeof(float), cudaMemcpyHostToDevice));
        //each process use 1 GPU
    int nDev = 1;

    //initializing GPU memery based on localRank
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(localRank * nDev + i));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    //generating NCCL unique ID at one process and broadcasting it to all
    ncclUniqueId id;
    if (myRank == 0)
    {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL, group API is required around ncclCommInitRank as it is called across multiple GPUs in each thread/process
    ncclComm_t comms[nDev];
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaSetDevice(localRank * nDev + i));
        NCCLCHECK(ncclCommInitRank(comms + i, nRanks * nDev, id, myRank * nDev + i));
    }
    NCCLCHECK(ncclGroupEnd());

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
        super_scaler_all_reduce_device(sendbuff[0], size, myRank, nRanks, localRank, comms, s);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / 10;
	std::cout << "test_device_nccl, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	
    //get gradients after allreduce
    for (int i = 0; i < size; i++)
    {
        gradients[i] = 0;
    }
    CUDACHECK(cudaMemcpy(gradients, sendbuff[0], sizeof(float) * size, cudaMemcpyDeviceToHost));

    
    //finalizing NCCL
    for (int i = 0; i < nDev; i++)
    {
        ncclCommDestroy(comms[i]);
    }

    //freeing device memory
    CUDACHECK(cudaFree(sendbuff[0]));

    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
        if(gradients[i] != i )
        {
            std::cout <<  "test_host fail " <<gradients[i] << " " << (nRanks)*(nRanks-1)/2 << "\n" ;
            break;
        }
    }
}

int main()
{
    int myRank = 0, nRanks = 0, localRank = 0;
    initialization(myRank, nRanks, localRank);

    //char hostname[1024];int length[1024];
    //MPI_Get_address(hostname);
    //std::cout << myRank << " " << hostname << std::endl;
/*
    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host mpi at " << myRank << std::endl;
    test_mpi_host(myRank, nRanks, localRank, 64000000);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host USR" << std::endl;
    test_USR_host(myRank, nRanks, localRank, 64000000);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host nccl" << std::endl;
    test_host(myRank, nRanks, localRank, 64000000);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_device nccl" << std::endl;
    test_device(myRank, nRanks, localRank, 64000000);
*/
    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_rdma at " << myRank << std::endl;
    test_rdma(myRank, nRanks, localRank, 64000000);

/*
    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host nccl" << std::endl;
    test_host(myRank, nRanks, localRank, 1000000);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host mpi" << std::endl;
    test_mpi_host(myRank, nRanks, localRank, 1000000);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host USR" << std::endl;
    test_USR_host(myRank, nRanks, localRank, 1000000);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_device nccl" << std::endl;
    test_device(myRank, nRanks, localRank, 1000000);
*/
    //std::cout << "=======================================================================" << std::endl;
    //std::cout << "test_rdma at " << myRank << std::endl;
    //test_rdma(myRank, nRanks, localRank, 1000000);
/*
    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_device 2nd" << std::endl;
    test_device(myRank, nRanks, localRank);
*/
    finalization();

    return 0;
}