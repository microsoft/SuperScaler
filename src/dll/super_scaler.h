#ifndef SUPER_SCALAR_H_
#define SUPER_SCALAR_H_

#include <cstdio>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "blas/blas.h"
#include "primitive_operation/comm_primitive.h"
#include "config_parse/parse.h"
#include "tools/check.h"

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

void MPI_usr_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank, plan plan);


#endif // SUPER_SCALAR_H_