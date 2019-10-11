#include "super_scaler.h"


void initialization(int &myRank, int &nRanks, int &localRank)
{
    //initializing MPI
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    global_cfg.parse_excution_plan("config/configure.cfg");

    //calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    for (int p = 0; p < nRanks; p++)
    {
        if (p == myRank)
        {
            break;
        }
        if (hostHashs[p] == hostHashs[myRank])
        {
            localRank++;
        }
    }
}

void finalization()
{
    //finalizing MPI
    MPICHECK(MPI_Finalize());
}


void MPI_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank)
{
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, (void *)gradients,
        (int) size,
        MPI_FLOAT,
        MPI_SUM,
        MPI_COMM_WORLD));
    
    for (int i = 0; i < size; i++)
    {
        gradients[i] /= nRanks;
    }
    //call back
    //(*callback)();
}

void MPI_usr_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank, plan plan)
{
    MpiCommPrimitive* mpicommprimitive_ = new MpiCommPrimitive();
    //mpicommprimitive_->getinfo();

    for (auto op_:plan.operation)
    {
        mpicommprimitive_->execute(gradients, size, myRank, nRanks, localRank, op_);
    }

    for (int i = 0; i < size; i++)
    {
        gradients[i] /= nRanks;
    }

    delete mpicommprimitive_;
}

