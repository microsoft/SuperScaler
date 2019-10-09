#include "super_scaler.h"


void initialization(int &myRank, int &nRanks, int &localRank)
{
    //initializing MPI
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    global_cfg.parse_excution_plan("configure/configure.cfg");

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

void MPI_usr_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank,
                                    plan plan, void* output_ptr)
{

    //auto plan = global_cfg.cfg_table["allreduce.classifier.6.bias"];

    MPI_Status recv_status;
    MPI_Request recv_req;
    //void* output_ptr = malloc(size*sizeof(float));
    float* output = (float*)output_ptr;

    for(auto op_ :plan.operation)
    {
        if(op_.operation_type == "send_receive")
        {
            if(op_.average)
            {
                float* segment_send = (float*)gradients + op_.send_address[myRank];
                float* segment_receive = (float*)gradients + op_.receive_address[myRank];
                float* segment_receive2 = output + op_.receive_address[myRank];
                MPI_Irecv(segment_receive2, op_.receive_length[myRank],
                        MPI_FLOAT, 
                        op_.receive_target[myRank], 
                        0, MPI_COMM_WORLD, &recv_req);
                MPI_Send(segment_send, op_.send_length[myRank],
                        MPI_FLOAT, 
                        op_.send_target[myRank], 
                        0, MPI_COMM_WORLD);

                MPI_Wait(&recv_req, &recv_status);

                for(int i = 0 ; i < op_.receive_length[myRank]; i++)
                    segment_receive[i] += segment_receive2[i];
            }
            else
            {
                float* segment_send = (float*)gradients + op_.send_address[myRank];
                float* segment_receive = (float*)gradients + op_.receive_address[myRank];
                MPI_Sendrecv(segment_send, op_.send_length[myRank],
                             MPI_FLOAT,
                             op_.send_target[myRank], 0,
                             segment_receive, op_.receive_length[myRank],
                             MPI_FLOAT,
                             op_.receive_target[myRank], 
                             0, MPI_COMM_WORLD, &recv_status);
            }
        }
        else if(op_.operation_type == "send")
          {
            if(op_.send_target[myRank] == -1)
              continue;
            else
            {
              float* segment_send = (float*)gradients + op_.send_address[myRank];
              MPI_Send(segment_send, op_.send_length[myRank],
                       MPI_FLOAT, 
                       op_.send_target[myRank], 0, MPI_COMM_WORLD);
            }
          }
          else if(op_.operation_type == "receive")
          {
            if(op_.receive_target[myRank] == -1)
              continue;
            else
            {
              if(op_.average){
                float* segment_receive = (float*)gradients + op_.receive_address[myRank];
                float* segment_receive2 = output + op_.receive_address[myRank];
                MPI_Irecv(segment_receive2, op_.receive_length[myRank],
                          MPI_FLOAT, 
                          op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_req);
                MPI_Wait(&recv_req, &recv_status);
                for(int i = 0 ; i < op_.receive_length[myRank]; i++)
                    segment_receive[i] += segment_receive2[i];
              }
              else
              {
                float* segment_receive = (float*)gradients + op_.receive_address[myRank];
                MPI_Recv(segment_receive, op_.receive_length[myRank],
                         MPI_FLOAT, 
                         op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_status);
              }
              
            }
          }
    }
    //delete[] output;
    
    for (int i = 0; i < size; i++)
    {
        gradients[i] /= nRanks;
    }
    //call back
    //(*callback)();
}

void nccl_super_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank,
                                  float **sendbuff, float **recvbuff, ncclComm_t* comms, cudaStream_t *s)
{
    //each process use 1 GPU
    int nDev = 1;

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaMemcpy(sendbuff[i], gradients, size * sizeof(float), cudaMemcpyHostToDevice));
    }
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
        CUDACHECK(cudaMemcpy(gradients, recvbuff[i], sizeof(float) * size, cudaMemcpyDeviceToHost));
    }

    //(*callback)();
}

void nccl_super_scaler_all_reduce_device(float *gradients, int size, int myRank, int nRanks, int localRank,
                                    ncclComm_t* comms, cudaStream_t *s)
{
    //each process use 1 GPU
    int nDev = 1;
    //calling NCCL communication API. Group API is required when using multiple devices per thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        NCCLCHECK(ncclAllReduce((const void *)gradients, (void *)gradients, size, ncclFloat, ncclSum, comms[i], s[i]));
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
        gradients_Average(gradients, size, nRanks);
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    
    //call back
    //(*callback)();
}
