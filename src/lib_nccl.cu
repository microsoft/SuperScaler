#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>

#include "nnscaler.h"

#define CUDACHECK(cmd) do {                         \
	  cudaError_t e = cmd;                              \
	  if( e != cudaSuccess ) {                          \
			    printf("Failed: Cuda error %s:%d '%s'\n",             \
						        __FILE__,__LINE__,cudaGetErrorString(e));   \
			    exit(EXIT_FAILURE);                             \
			  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
	  ncclResult_t r = cmd;                             \
	  if (r!= ncclSuccess) {                            \
			    printf("Failed, NCCL error %s:%d '%s'\n",             \
						        __FILE__,__LINE__,ncclGetErrorString(r));   \
			    exit(EXIT_FAILURE);                             \
			  }                                                 \
} while(0)


void test_nccl(){
		int size = 32*1024*1024;
		int myRank, nRanks, localRank = 0;
		myRank = 0; nRanks = 1;


	  ncclUniqueId id;
		ncclComm_t comm;
		float *sendbuff, *recvbuff;
		cudaStream_t s;
		
		if (myRank == 0)
			ncclGetUniqueId(&id);

		//picking a GPU based on localRank, allocate device buffers
		cudaSetDevice(localRank);
		cudaMalloc(&sendbuff, size * sizeof(float));
		cudaMalloc(&recvbuff, size * sizeof(float));
		cudaStreamCreate(&s);

		ncclCommInitRank(&comm, nRanks, id, myRank);

		//communicating using NCCL
		NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
				comm, s));


		//completing NCCL operation by synchronizing on the CUDA stream
		CUDACHECK(cudaStreamSynchronize(s));


		//free device buffers
		CUDACHECK(cudaFree(sendbuff));
		CUDACHECK(cudaFree(recvbuff));


		//finalizing NCCL
		ncclCommDestroy(comm);
}

