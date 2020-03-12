#include "super_scaler.h"
#include <map>

static RdmaCommPrimitive *rdmaCommPrimitive = NULL;
static MpiCommPrimitive *mpiCommPrimitive = NULL;
static PlanTable table;
static std::map<std::string, int> mpiRankListHostRank;
static std::map<int, std::string> mpiRankListRankHost;

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

void initialization(int &myRank, int &nRanks, int &localRank)
{
    // initializing MPI
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

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

    // load plan
    std::string planPath = "plan/execution_plan/" + std::to_string(myRank) + ".cfg";
    table.readConfig(planPath);

    // mpiCommPrimitive initialization
    mpiCommPrimitive = new MpiCommPrimitive();
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // MPI rank list initialization
    int mpiRankDataSize = 20;  // len("0:10.0.0.21 0 12001;") = 20;
    unsigned char *mpiRankData = new unsigned char[mpiRankDataSize * nRanks];
    MPIRankListInitialization(mpiRankData, mpiRankDataSize, myRank, nRanks);
    // for(auto iter = mpiRankListHostRank.begin(); iter != mpiRankListHostRank.end(); iter++) 
    // {
    //     std::cout << iter->first << " : " << iter->second << std::endl;
    // }

    // RDMA initialization
    // Plan plan = table.getFirstAllreducePlan();
    // std::vector<std::string> endPoints = plan.getEndPoints();
    std::vector<std::string> endPoints;
    for (int i = 0; i < nRanks; i++)
    {
        endPoints.push_back(mpiRankListRankHost[i]);
    }
    int endPointsCount = endPoints.size();

    std::vector<std::string> ips;
    std::vector<std::string> ports;
    for (int i = 0; i < endPointsCount; i++)
    {
        std::string endPoint = endPoints[i];
        int blankIndex = endPoint.find(" ");
        std::string ip = endPoint.substr(0, blankIndex);
        std::string gpu = endPoint.substr(blankIndex + 1, 1);
        std::string port = endPoint.substr(blankIndex + 3, endPoints[i].size() - (blankIndex + 3) + 1);
        ips.push_back(ip);
        ports.push_back(port);
    }

    rdmaCommPrimitive = new RdmaCommPrimitive();
    rdmaCommPrimitive->initialization(ips, ports, myRank, nRanks, localRank, 512 * 1024 * 1024);
}

void MPIRankListInitialization(unsigned char *mpiRankData, int mpiRankDataSize, int myRank, int nRanks)
{
    for (int i = 0; i < mpiRankDataSize * nRanks; i++)
    {
        mpiRankData[i] = ' ';
    }

    std::string selfName = table.getSelfName();
    int mpiRankDataStartIndex = myRank * mpiRankDataSize;
    mpiRankData[mpiRankDataStartIndex] = myRank + '0';
    mpiRankData[mpiRankDataStartIndex + 1] = ':';
    mpiRankDataStartIndex += 2;
    for (int i = mpiRankDataStartIndex; i < mpiRankDataStartIndex + selfName.size(); i++)
    {
        mpiRankData[i] = selfName[i - mpiRankDataStartIndex];
    }
    mpiRankData[mpiRankDataStartIndex + selfName.size()] = ';';

    int sendTarget = (myRank + 1) % nRanks;
    int receiveTarget = (myRank + nRanks - 1) % nRanks;

    int sendIndex = myRank;
    int receiveIndex = (myRank + nRanks - 1) % nRanks;

    for (int i = 0; i < nRanks - 1; i++)
    {
        int sendAddress = sendIndex * mpiRankDataSize;
        if (sendIndex == 0)
        {
            sendIndex = nRanks - 1;
        }
        else
        {
            sendIndex--;
        }

        int receiveAddress = receiveIndex * mpiRankDataSize;
        if (receiveIndex == 0)
        {
            receiveIndex = nRanks - 1;
        }
        else
        {
            receiveIndex--;
        }

        MPI_Send(mpiRankData + sendAddress, mpiRankDataSize, MPI_UNSIGNED_CHAR, sendTarget, 0, MPI_COMM_WORLD);
        MPI_Status recv_status;
        MPI_Recv(mpiRankData + receiveAddress, mpiRankDataSize, MPI_UNSIGNED_CHAR, receiveTarget, 0, MPI_COMM_WORLD, &recv_status);
    }
    for (int i = 0; i < nRanks - 1; i++)
    {
        int sendAddress = sendIndex * mpiRankDataSize;
        if (sendIndex == 0)
        {
            sendIndex = nRanks - 1;
        }
        else
        {
            sendIndex--;
        }

        int receiveAddress = receiveIndex * mpiRankDataSize;
        if (receiveIndex == 0)
        {
            receiveIndex = nRanks - 1;
        }
        else
        {
            receiveIndex--;
        }

        MPI_Send(mpiRankData + sendAddress, mpiRankDataSize, MPI_UNSIGNED_CHAR, sendTarget, 0, MPI_COMM_WORLD);
        MPI_Status recv_status;
        MPI_Recv(mpiRankData + receiveAddress, mpiRankDataSize, MPI_UNSIGNED_CHAR, receiveTarget, 0, MPI_COMM_WORLD, &recv_status);
    }

    std::string mpiRankDataResult(reinterpret_cast<char*>(mpiRankData));
    for (int i = 0; i < nRanks; i++)
    {
        int startIndex = mpiRankDataSize * i;
        int rank = mpiRankData[startIndex] - '0';
        startIndex += 2;  // for "0:"

        int endIndex = startIndex;
        while (mpiRankData[endIndex] != ';')
        {
            endIndex++;
        }

        int length = endIndex - startIndex;  // except for ';'
        std::string tmpStr = mpiRankDataResult.substr(startIndex, length);

        mpiRankListHostRank[tmpStr] = rank;
        mpiRankListRankHost[rank] = tmpStr;
    }
}

void finalization()
{
    MPICHECK(MPI_Finalize());
}

int allReduce(float *gradients, size_t size, std::string tensorName)
{
    if (!table.hasPlan(tensorName))
    {
        std::cout << "Error: " + tensorName + "not exist in plan" << std::endl;
        return 0;
    }

    std::string selfName = table.getSelfName();
    Plan plan = table.getPlan(tensorName);
    // plan.displayPlan();

    OperationType OT = plan.getOperationType();
    if (OT == allreduceOT)
    {
        if (true) // algorithm == Ring, describe in plan
        {
            allReduce_Ring(gradients, size, selfName, plan);
        }
    }
    else
    {
        std::cout << "Error: " + tensorName + "is not allreduce opreation" << std::endl;
        return 0;
    }

    return 1;
}

int sendReceive(unsigned char *data, size_t size, std::string tensorName)
{
    if (!table.hasPlan(tensorName))
    {
        std::cout << "Error: " + tensorName + "not exist in plan" << std::endl;
        return 0;
    }

    std::string selfName = table.getSelfName();
    Plan plan = table.getPlan(tensorName);
    // plan.displayPlan();

    OperationType OT = plan.getOperationType();
    if (OT == sendOT)
    {
        send_MPI(data, size, selfName, plan);
    }
    else if (OT == receiveOT)
    {
        receive_MPI(data, size, selfName, plan);
    }
    else
    {
        std::cout << "Error: " + tensorName + "is not allreduce opreation" << std::endl;
        return 0;
    }

    return 1;
}

void send_MPI(unsigned char *data, size_t size, std::string selfName, Plan plan)
{
    std::vector<std::string> endPoints = plan.getEndPoints();
    int endPointsCount = endPoints.size();
    for (int i = 0; i < endPointsCount; i++)
    {
        std::string tmp = endPoints[i];
        int sendTarget = mpiRankListHostRank[tmp];
        mpiCommPrimitive->send(data, sendTarget, 0, size);
    }
}

void receive_MPI(unsigned char *data, size_t size, std::string selfName, Plan plan)
{
    std::vector<std::string> endPoints = plan.getEndPoints();
    int endPointsCount = endPoints.size();
    for (int i = 0; i < endPointsCount; i++)
    {
        std::string tmp = endPoints[i];
        int receiveTarget = mpiRankListHostRank[tmp];
        mpiCommPrimitive->recieve(data, receiveTarget, 0, size);
    }
}

void allReduce_Ring(float *gradients, size_t size, std::string selfName, Plan plan)
{
    std::vector<std::string> endPoints = plan.getEndPoints();
    int endPointsCount = endPoints.size();
    int rank = 0;
    for (int i = 0; i < endPointsCount; i++)
    {
        if (endPoints[i] == selfName)
        {
            break;
        }
        rank++;
    }

    CommunicationType CT = plan.getCommunicationType();

    if (CT == DefaultCT || CT == RdmaCT)
    {
        rdmaCommPrimitive->RDMA_Register_GPU_MemRegion(gradients, size);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    int sendTarget = (rank + 1) % endPointsCount;
    int receiveTarget = (rank + endPointsCount - 1) % endPointsCount;
    size_t chunkSize = size / endPointsCount;

    int sendIndex = rank;
    int receiveIndex = (rank + endPointsCount - 1) % endPointsCount;

    // scatter reduce
    for (int i = 0; i < endPointsCount - 1; i++)
    {
        int sendAddress = sendIndex * chunkSize;
        if (sendIndex == 0)
        {
            sendIndex = endPointsCount - 1;
        }
        else
        {
            sendIndex--;
        }

        int receiveAddress = receiveIndex * chunkSize;
        if (receiveIndex == 0)
        {
            receiveIndex = endPointsCount - 1;
        }
        else
        {
            receiveIndex--;
        }

        float *receiveBuffer;
        if (CT == DefaultCT || CT == RdmaCT)
        {
            receiveBuffer = allReduce_Transmit_RDMA(rdmaCommPrimitive, gradients, sendTarget, sendAddress, chunkSize, receiveTarget, receiveAddress, chunkSize);
        }
        allReduce_Gradients_SumOrCover(gradients, receiveAddress, receiveBuffer, chunkSize, 0);
    }

    // allgather
    for (int i = 0; i < endPointsCount - 1; i++)
    {
        int sendAddress = sendIndex * chunkSize;
        if (sendIndex == 0)
        {
            sendIndex = endPointsCount - 1;
        }
        else
        {
            sendIndex--;
        }

        int receiveAddress = receiveIndex * chunkSize;
        if (receiveIndex == 0)
        {
            receiveIndex = endPointsCount - 1;
        }
        else
        {
            receiveIndex--;
        }

        float *receiveBuffer;
        if (CT == DefaultCT || CT == RdmaCT)
        {
            receiveBuffer = allReduce_Transmit_RDMA(rdmaCommPrimitive, gradients, sendTarget, sendAddress, chunkSize, receiveTarget, receiveAddress, chunkSize);
        }
        allReduce_Gradients_SumOrCover(gradients, receiveAddress, receiveBuffer, chunkSize, 1);
    }

    gradients_Average(gradients, size, endPointsCount);
}

float *allReduce_Transmit_RDMA(RdmaCommPrimitive *rdmaCommPrimitive, float *gradients, int sendTarget, int sendAddress, int sendLength, int receiveTarget, int receiveAddress, int receiveLength)
{
    float *receiveBuffer = rdmaCommPrimitive->send_receive(gradients, sendTarget, sendAddress, sendLength, receiveTarget, receiveAddress, receiveLength);
    cudaDeviceSynchronize();
    return receiveBuffer;
}

void allReduce_Gradients_SumOrCover(float *gradients, int receiveAddress, float *receiveBuffer, int receiveLength, int type)
{
    float *gradientsOffset = gradients + receiveAddress;
    if (type == 0)
    {
        gradients_Reduce(gradientsOffset, receiveBuffer, receiveLength);
    }
    else if (type == 1)
    {
        cudaMemcpy(gradientsOffset, receiveBuffer, receiveLength * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
}
