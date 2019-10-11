#include "nccl.h"

void nccl_init(const int &myRank, const int &nRanks, const int &localRank, const size_t &size,
                const int &nDev, float ** &sendbuff, float ** &recvbuff, cudaStream_t *s, ncclComm_t *comms){

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(localRank * nDev + i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    ncclUniqueId id;
    if (myRank == 0){
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        NCCLCHECK(ncclCommInitRank(comms + i, nRanks * nDev, id, myRank * nDev + i));
    }
    NCCLCHECK(ncclGroupEnd());

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

void nccl_finalization(float ** &sendbuff, float ** &recvbuff, ncclComm_t *comms, const int &nDev){
    //freeing device memory
    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        ncclCommDestroy(comms[i]);
    }
}

void execute_nccl_all_reduce(std::string mode, float **sendbuff, float **recvbuff,int size, int myRank, int nRanks, int localRank,
                                        ncclComm_t* comms, cudaStream_t *s, const int & nDev, float *gradients)
{
    if(mode == "host"){
        for (int i = 0; i < nDev; ++i)
        {
            CUDACHECK(cudaMemcpy(sendbuff[i], gradients, size * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    nccl_super_scaler_all_reduce_device(sendbuff, recvbuff, size, myRank, nRanks, localRank, comms, s, nDev);
    
    if(mode == "host"){
        for (int i = 0; i < nDev; ++i)
        {
            CUDACHECK(cudaMemcpy(gradients, recvbuff[i], sizeof(float) * size, cudaMemcpyDeviceToHost));
        }
    }
}

void nccl_super_scaler_all_reduce_host(float *gradients, float **sendbuff, float **recvbuff, int size, int myRank, int nRanks, int localRank,
                                        ncclComm_t* comms, cudaStream_t *s, const int & nDev)
{
    //each process use 1 GPU
     for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaMemcpy(sendbuff[i], gradients, size * sizeof(float), cudaMemcpyHostToDevice));
    }
    //calling NCCL communication API. Group API is required when using multiple devices per thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        NCCLCHECK(ncclAllReduce((const void *) sendbuff[i], (void *) recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA stream to complete NCCL communication
    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    //get gradients after allreduce
    for (int i = 0; i < nDev; i++)
    {
        gradients_Average(recvbuff[i], size, nRanks);
        CUDACHECK(cudaStreamSynchronize(s[i]));

        CUDACHECK(cudaMemcpy(gradients, recvbuff[i], sizeof(float) * size, cudaMemcpyDeviceToHost));
    }
}

void nccl_super_scaler_all_reduce_device(float **sendbuff, float **recvbuff, int size, int myRank, int nRanks, int localRank,
                                         ncclComm_t* comms, cudaStream_t *s, const int & nDev)
{
    //each process use 1 GPU
    //calling NCCL communication API. Group API is required when using multiple devices per thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    //synchronizing on CUDA stream to complete NCCL communication
    for (int i = 0; i < nDev; i++)
    {
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    //get gradients after allreduce
    for (int i = 0; i < nDev; i++)
    {
        //CUDACHECK(cudaSetDevice(localRank * nDev + i));
        gradients_Average(recvbuff[i], size, nRanks);
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }
}
