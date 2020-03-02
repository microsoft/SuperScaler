#include "super_scaler.h"

static RdmaCommPrimitive *rdmaCommPrimitive = NULL;

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

void initialization(int &myRank, int &nRanks, int &localRank, PlanTable &table)
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

    std::string planPath = "plan/execution_plan/" + std::to_string(myRank) + ".cfg";
    table.readConfig(planPath);

    Plan plan = table.getFirstAllreducePlan();
    std::vector<std::string> endPoints = plan.getEndPoints();
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
    rdmaCommPrimitive->initialization(ips, ports, myRank, nRanks, localRank, 16 * 1024 * 1024);

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

void finalization()
{
    MPICHECK(MPI_Finalize());
}

int allReduce(float *gradients, size_t size, std::string tensorName, PlanTable table)
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
    if (OT != allreduceOT)
    {
        std::cout << "Error: " + tensorName + "is not allreduce opreation" << std::endl;
        return 0;
    }

    if (true) // algorithm == Ring, describe in plan
    {
        allReduce_Ring(gradients, size, selfName, plan);
    }

    return 1;
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
