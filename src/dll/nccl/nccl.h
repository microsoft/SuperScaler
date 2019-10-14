#include "../blas/blas.h"
#include "../tools/check.h"
#include <nccl.h>

void nccl_init(const int &myRank, const int &nRanks, const int &localRank, const size_t &size, const int &nDev,
                 float ** &sendbuff, float ** &recvbuff, cudaStream_t *s, ncclComm_t *comms);

void nccl_finalization(float ** &sendbuff, float ** &recvbuff, ncclComm_t *comms, const int &nDev);

void nccl_super_scaler_all_reduce_device(float **& sendbuff, float **& recvbuff, int size, int myRank, int nRanks, int localRank,
                                         ncclComm_t* comms, cudaStream_t *s, const int & nDev);

void execute_nccl_all_reduce(std::string mode, float ** &sendbuff, float ** &recvbuff, int size, int myRank, int nRanks, int localRank,
                                        ncclComm_t* comms, cudaStream_t *s, const int & nDev, float * &gradients);