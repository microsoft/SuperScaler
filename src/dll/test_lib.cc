#include <stdio.h>
#include <iostream>
#include <chrono>
#include "super_scaler.h"
#include "rdma/rdma.h"
#include "config_parse/parse.h"

const int test_times = 1;
const size_t maxerrcount = 10;

RdmaCommPrimitive * rdmaCommPrimitive_ = new RdmaCommPrimitive();

void gradients_init(float* &gradients, const size_t &size){
    for (int i = 0; i < size; i++){
        gradients[i] = i;
    }
}

void callBackFunction()
{
    std::cout << "Call Back Success!" << std::endl;
}

void test_nccl_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);

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
    for(int i = 0; i < test_times; i++)
        nccl_super_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank,
                                     sendbuff, recvbuff, comms, s);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
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
	    float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout <<  "test_host fail " <<gradients[i] << " != " << target_value << "\n" ;
            break;
        }
    }

    delete [] gradients;
}

void test_nccl_device(int myRank, int nRanks, int localRank, size_t size)
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
    //initializing GPU memery based on localRan
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
    for(int i = 0; i < test_times; i++)
        nccl_super_scaler_all_reduce_device(sendbuff[0], size, myRank, nRanks, localRank, comms, s);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
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
	    float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout <<  "test_host fail " <<gradients[i] << " != " << target_value << "\n" ;
            break;
        }
    }

	delete [] gradients;
}

void test_rdma_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);
    rdmaCommPrimitive_->RDMA_Register_CPU_MemRegion(gradients, size);

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < test_times; i++)
    {
        auto plan = rdmaCommPrimitive_->get_rdma_cfg().cfg_table["allreduce.classifier.6.bias"];

        for(auto op_:plan.operation)
        {										  
            rdmaCommPrimitive_->execute(gradients, size, myRank, nRanks, localRank, op_);
        }
        for (int k = 0; k < size; k++){
            gradients[k] /= nRanks;
        }
    }
    //    RDMA_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
    std::cout << "test_rdma_host, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
    std::cout << "After all reduce" << std::endl;
																   
    size_t errcount = 0;
    for (int i = 0; i < size; i++)
    {
        float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout << "###  test_rdma_host fail " << gradients[i] << " != " << target_value << "\n";
            if (++errcount > maxerrcount)
                break;
        }
    }

    delete [] gradients;
    //cudaFreeHost(gradients); //TODO fixme
}

void test_rdma_device(int myRank, int nRanks, int localRank, size_t size) //test_rdma by interface
{
    float *gradients = new float[size];
    gradients_init(gradients,size);

    float *sendbuff = nullptr; 

    CUDACHECK(cudaSetDevice(localRank * 1 + 0)); //TODO
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff, gradients, size * sizeof(float), cudaMemcpyHostToDevice));

    rdmaCommPrimitive_->RDMA_Register_GPU_MemRegion(sendbuff, size);    
    
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < test_times; i++)
    {
        auto plan = rdmaCommPrimitive_->get_rdma_cfg().cfg_table["allreduce.classifier.6.bias"];

        for(auto op_:plan.operation)
        {
            if(op_.operation_type == "write")
            {
                rdmaCommPrimitive_->run_write_device(sendbuff, size, myRank, nRanks, localRank, op_);
            }
        }
        gradients_Average(sendbuff, size, nRanks);
        cudaDeviceSynchronize();
    }
    //    RDMA_scaler_all_reduce_device(sendbuff, size, myRank, nRanks, localRank);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
    std::cout << "test_rdma_device, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";

    CUDACHECK(cudaMemcpy(gradients, sendbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "After all reduce" << std::endl;
    size_t errcount = 0;
    for (int i = 0; i < size; i++)
    {
        float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout << "### test_rdma_device fail " << gradients[i] << " != " << target_value << "\n";
            if (++errcount > maxerrcount)
                break;
        }
    }

    CUDACHECK(cudaFree(sendbuff));
    delete [] gradients;
    //cudaFreeHost(gradients); //TODO fixme
}


void test_mpi_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);

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
	    float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout <<  "test_host fail " <<gradients[i] << "!= " << target_value << "\n" ;
            break;
        }
    }
    
    delete [] gradients;				
}

void test_mpi_USR_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);
	
    auto plan = global_cfg.cfg_table["allreduce.classifier.6.bias"];
																	
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
        MPI_usr_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank, plan);
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / 10;
	std::cout << "test_mpi_host, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";																																																					  
    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
	    float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout <<  "test_host fail " << gradients[i] << " !=  " << target_value << "\n" ;
            break;
        }
    }

    delete [] gradients;
}

inline void print_cpu_array(float* array, std::string name = "", size_t size = 10){
    std::cout << "print_cpu_array (" << name << "): ";
    for (size_t i = 0; i < size; i++)
        std::cout << ", " << array[i];
    std::cout << std::endl;
}

inline void print_gpu_array(float* array, std::string name = "", size_t size = 10){
    float *tmp = (float *)malloc(sizeof(float) * size);
    CUDACHECK(cudaMemcpy(tmp, array, sizeof(float) * size, cudaMemcpyDeviceToHost));
    std::cout << "print_gpu_array (" << name << "): ";
    for (size_t i = 0; i < size; i++)
        std::cout << ", " << tmp[i];
    std::cout << std::endl;
    free(tmp);
}

int main()
{
    int myRank = 0, nRanks = 0, localRank = 0;
    initialization(myRank, nRanks, localRank);

    // std::cout << "=======================================================================" << std::endl;
    // std::cout << "test_host mpi at " << myRank << std::endl;
    // test_mpi_host(myRank, nRanks, localRank, 64*1024*1024);
/*
    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host USR" << std::endl;
    test_mpi_USR_host(myRank, nRanks, localRank, 16*1024*1024);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_host nccl" << std::endl;
    test_nccl_host(myRank, nRanks, localRank, 16*1024*1024);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_device nccl" << std::endl;
    test_nccl_device(myRank, nRanks, localRank, 16*1024*1024);
*/															
    rdmaCommPrimitive_->set_cfg_RDMA(global_cfg, myRank, nRanks, localRank, 16*1024*1024);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_rdma at " << myRank << std::endl;
    test_rdma_host(myRank, nRanks, localRank, 16*1024*1024);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_rdma at " << myRank << std::endl;
    test_rdma_device(myRank, nRanks, localRank, 16*1024*1024);
/*
    finallize_RDMA(myRank, nRanks, localRank);
*/
    finalization();

    return 0;
}
