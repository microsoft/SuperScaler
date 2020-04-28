#include "comm_primitive.h"

const int test_times = 10;
const size_t maxerrcount = 10;
const int nDev = 1;

CfgTable global_cfg;
RdmaCommPrimitive * rdmaCommPrimitive_ = new RdmaCommPrimitive();

static uint64_t getHostHash(const char *string)
{
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

void initialization(int &myRank, int &nRanks, int &localRank)
{
    //initializing MPI
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    global_cfg.parse_excution_plan("config/configure.cfg");

    //calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    for (int p = 0; p < nRanks; p++)
    {
        if (p == myRank)
        {
            break;
        }
        if (hostHashs[p] == hostHashs[myRank])
        {
            localRank++;
        }
    }
}

void finalization()
{
    //finalizing MPI
    MPICHECK(MPI_Finalize());
}

void MPI_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank)
{
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, (void *)gradients,
        (int) size,
        MPI_FLOAT,
        MPI_SUM,
        MPI_COMM_WORLD));
    
    for (int i = 0; i < size; i++)
    {
        gradients[i] /= nRanks;
    }
    //call back
    //(*callback)();
}

void MPI_usr_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank, plan plan)
{
    MpiCommPrimitive* mpicommprimitive_ = new MpiCommPrimitive();
    //mpicommprimitive_->getinfo();

    for (auto op_:plan.operation)
    {
        mpicommprimitive_->execute(gradients, size, myRank, nRanks, localRank, op_);
    }

    for (int i = 0; i < size; i++)
    {
        gradients[i] /= nRanks;
    }

    delete mpicommprimitive_;
}

void gradients_init(float* &gradients, const size_t &size){
    for (int i = 0; i < size; i++){
        gradients[i] = i;
    }
}

void gradients_check(float* &gradients, const size_t &size){
    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
	    float target_value = i;
        if (std::fabs(gradients[i] - target_value) > 0.0001)
        {
            std::cout <<  "test_host fail " <<gradients[i] << " != " << target_value << "\n" ;
            return;
        }
    }

    std::cout << "all reduce success!" << std::endl;
}

void test_nccl_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);

    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);
    ncclComm_t comms[nDev];

    nccl_init(myRank, nRanks, localRank, size, nDev, sendbuff, recvbuff, s, comms);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < test_times; i++)
    {
        execute_nccl_all_reduce("host", sendbuff, recvbuff, size, myRank, nRanks, localRank,
                                        comms, s, nDev, gradients);
    }
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
	std::cout << "test_host_nccl, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";

    nccl_finalization(sendbuff, recvbuff, comms, nDev);
    gradients_check(gradients, size);

    delete [] gradients;
}

void test_nccl_device(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients, size);
    
    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));

    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);
    ncclComm_t comms[nDev];

    nccl_init(myRank, nRanks, localRank, size, nDev, sendbuff, recvbuff, s, comms);

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaMemcpy(sendbuff[i], gradients, size * sizeof(float), cudaMemcpyHostToDevice));
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < test_times; i++)
    {
        execute_nccl_all_reduce("device", sendbuff, recvbuff, size, myRank, nRanks, localRank,
                                        comms, s, nDev, gradients);
    }

    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
	std::cout << "test_device_nccl, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";

    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaMemcpy(gradients, recvbuff[i], sizeof(float) * size, cudaMemcpyDeviceToHost));
    }

    nccl_finalization(sendbuff, recvbuff, comms, nDev);
    gradients_check(gradients, size);

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
																   
    gradients_check(gradients, size);

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
    gradients_check(gradients, size);
    CUDACHECK(cudaFree(sendbuff));
    delete [] gradients;
    //cudaFreeHost(gradients); //TODO fixme
}


void test_mpi_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);
	
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++)
    {
        MPI_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank);
    }
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / 10;
	std::cout << "test_mpi_host, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
	
    gradients_check(gradients, size);
   
    delete [] gradients;				
}

void test_mpi_USR_host(int myRank, int nRanks, int localRank, size_t size)
{
    float *gradients = new float[size];
    gradients_init(gradients,size);

    auto plan = global_cfg.cfg_table["allreduce.classifier.6.bias"];
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::system_clock::now();
    //for(int i = 0; i < test_times; i++)
    std::cout << "start communication" << std::endl;

    MPI_usr_scaler_all_reduce_host(gradients, size, myRank, nRanks, localRank, plan);
    
    std::cout << "finish communication" << std::endl;
    
    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time) / test_times;
	std::cout << "test_mpi_host, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";

    gradients_check(gradients, size);
    delete [] gradients;
}



int main()
{
    int myRank = 0, nRanks = 0, localRank = 0;
    initialization(myRank, nRanks, localRank);

    // std::cout << "=======================================================================" << std::endl;
    // std::cout << "test_host mpi at " << myRank << std::endl;
    // test_mpi_host(myRank, nRanks, localRank, 64*1024*1024);

    // std::cout << "=======================================================================" << std::endl;
    // std::cout << "test_host USR" << std::endl;
    // test_mpi_USR_host(myRank, nRanks, localRank, 16*1024*1024);

    // std::cout << "=======================================================================" << std::endl;
    // std::cout << "test_host nccl" << std::endl;
    // test_nccl_host(myRank, nRanks, localRank, 16*1024*1024);
 
    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_device nccl" << std::endl;
    test_nccl_device(myRank, nRanks, localRank, 128 * 1024 * 1024);
															
    // rdmaCommPrimitive_->set_cfg_RDMA(myRank, nRanks, localRank, 16*1024*1024);

    // std::cout << "=======================================================================" << std::endl;
    // std::cout << "test_rdma at " << myRank << std::endl;
    // test_rdma_host(myRank, nRanks, localRank, 16*1024*1024);

    // std::cout << "=======================================================================" << std::endl;
    // std::cout << "test_rdma at " << myRank << std::endl;
    // test_rdma_device(myRank, nRanks, localRank, 16*1024*1024);

    // finallize_RDMA(myRank, nRanks, localRank);

    finalization();

    return 0;
}
