
//void test_nccl();
void test_nccl(int myRank, int nRanks, int rankPerNode);

//MPI based func
void init_mpi(int argc, char** argv, int &myRank, int &nRanks);
void close_mpi();
void test_mpi(int argc, char** argv);
void broadcast_meta(void* buffer, int size); 
