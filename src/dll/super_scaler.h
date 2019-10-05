#ifndef SUPER_SCALAR_H_
#define SUPER_SCALAR_H_

#include <cstdio>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "blas.h"

#include "parse.h"

CfgTable global_cfg;


auto start_time = std::chrono::system_clock::now();

void setup_time()
{
    start_time = std::chrono::system_clock::now();
}

void show_time()
{
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start_time;
	std::cout << "------------elapsed time: " << elapsed_seconds.count() <<'\n';
	
}

void show_time(std::string s)
{
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start_time;
	std::cout << s << "------------elapsed time: " << elapsed_seconds.count() <<'\n';	
}

#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess)                                  \
        {                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

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

void initialization(int &myRank, int &nRanks, int &localRank);

void finalization();

void MPI_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank);

void MPI_usr_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank,
                                    plan plan, void* output_ptr);

void super_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank,
                                  float **sendbuff, float **recvbuff, ncclComm_t* comms, cudaStream_t *s);

void super_scaler_all_reduce_device(float *gradients, int size, int myRank, int nRanks, int localRank, 
                                    ncclComm_t* comms, cudaStream_t *s);


#endif // SUPER_SCALAR_H_