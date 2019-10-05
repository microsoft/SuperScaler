#ifndef RDMA_H_
#define RDMA_H_

//#include "super_scaler.h"
#include <mpi.h>
#include "blas.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include "gdrapi.h"

#include "rdma_device_manager.h"

#include <iostream>
#include <mutex>
#include <condition_variable>
#include "parse.h"

#include <chrono>
#include <vector>

void set_cfg_RDMA(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients, float *gradients_gpu, float *buf_gpu, size_t size);
void RDMA_scaler_all_reduce_host(float *gradients, size_t size, int myRank, int nRanks, int localRank);
void RDMA_scaler_all_reduce_device(float *gradients, size_t size, int myRank, int nRanks, int localRank);
#endif