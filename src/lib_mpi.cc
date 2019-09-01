#include <mpi.h>
#include <iostream>

#include "nnscaler.h"

void test_mpi(int argc, char** argv){
  //initializing MPI
 int myRank = -1;
 int nRanks = -1;
 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
 MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
 
 std::cout << "rank = (" << myRank << ", " << nRanks << ")" << std::endl;

 MPI_Finalize();
}

