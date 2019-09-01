#include "nnscaler.h"

int main(int argc, char** argv) {
	//test_mpi(argc, argv);
	//test_nccl();
	int myRank = -1;
	int nRanks = -1;
	init_mpi(argc, argv, myRank, nRanks);
	test_nccl(myRank, nRanks, 1);
	close_mpi();
	return 0;
}
