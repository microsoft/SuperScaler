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



//int get_ncclUniqueId_size(){
//	return sizeof(ncclUniqueId);
//}
//
//int gen_ncclUniqueId(void* pid){
//  
//}
//
//void init_nccl(int myRank = 0, int nRanks = 1, int rankPerNode = 1){
//
//}

void test_nccl(int myRank = 0, int nRanks = 1, int rankPerNode = 1){
		int size = 32*1024*1024;
		int localRank = myRank % rankPerNode;

	  ncclUniqueId id;
		ncclComm_t comm;
		float *cpusendbuff = new float[size];
		float *cpurecvbuff = new float[size];
		for (int i = 0; i < size; i++)
			cpusendbuff[i] = myRank + 0.1;

		//
		for (int i = 0; i < size && i < 4; i++)
			printf("%f, ", cpusendbuff[i]);
		printf("\n");

		float *sendbuff, *recvbuff;
		cudaStream_t s;
		
		if (myRank == 0)
			ncclGetUniqueId(&id);

		//extra comm. library
		broadcast_meta((void*)(&id), sizeof(ncclUniqueId));

		//picking a GPU based on localRank, allocate device buffers
		cudaSetDevice(localRank);
		cudaMalloc(&sendbuff, size * sizeof(float));
		cudaMalloc(&recvbuff, size * sizeof(float));
		cudaStreamCreate(&s);

		cudaMemcpy((void*)sendbuff, (const void*)cpusendbuff, size * sizeof(float), cudaMemcpyHostToDevice);

		ncclCommInitRank(&comm, nRanks, id, myRank);

		//communicating using NCCL
		NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
				comm, s));


		//completing NCCL operation by synchronizing on the CUDA stream
		CUDACHECK(cudaStreamSynchronize(s));

		cudaMemcpy((void*)cpurecvbuff, (const void*)recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost);

		//free device buffers
		CUDACHECK(cudaFree(sendbuff));
		CUDACHECK(cudaFree(recvbuff));


		for (int i = 0; i < size && i < 4; i++)
			printf("%f, ", cpurecvbuff[i]);
		printf("\n");

		delete[] cpusendbuff;
		delete[] cpurecvbuff;

		//finalizing NCCL
		ncclCommDestroy(comm);
}

