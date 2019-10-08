#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <stdint.h>

#include "nnscaler.h"

#define MPICHECK(cmd) do {                          \
	int e = cmd;                                      \
	if( e != MPI_SUCCESS ) {                          \
		printf("Failed: MPI error %s:%d '%d'\n",        \
			__FILE__,__LINE__, e);   \
		exit(-1);                             \
	}                                                 \
} while(0) //exit(EXIT_FAILURE);

void broadcast_meta(void* buffer, int size) 
{
	MPICHECK(MPI_Bcast(buffer, size, MPI_BYTE, 0, MPI_COMM_WORLD));
}

void init_mpi(int argc, char** argv, int &myRank, int &nRanks){
	//initializing MPI
	myRank = -1;
	nRanks = -1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

	std::cout << "rank = (" << myRank << ", " << nRanks << ")" << std::endl;
}

void close_mpi(){
	MPI_Finalize();
}

void test_mpi(int argc, char** argv){
	int myRank = -1;
	int nRanks = -1;
	init_mpi(argc, argv, myRank, nRanks);
	close_mpi();
}

