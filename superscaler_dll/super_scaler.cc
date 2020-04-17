#include <map>
#include "super_scaler.h"

static PlanTable table;
cudaStream_t cudaStream;

static MpiCommPrimitive *mpiCommPrimitive = NULL;
static std::map<std::string, int> mpiRankListHostRank;
static std::map<int, std::string> mpiRankListRankHost;

static RdmaCommPrimitive *rdmaCommPrimitive = NULL;
float *rdmaSendBuff = nullptr;

const char * pipe_name = "test_cuda_comm_primitive";
static SharedPipe * sharedPipe = NULL;
static CudaIPCCommPrimitive * cudaIPCCommPrimitive = NULL;
float *cudaIpcRevBuff = nullptr;

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
    // MPI initialization -- myRank, nRanks
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // MPI initialization -- localRank
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

    // Plan initialization
    std::string planPath = "plan/execution_plan/" + std::to_string(myRank) + ".cfg";
    table.readConfig(planPath);

    // mpiCommPrimitive initialization
    mpiCommPrimitive = new MpiCommPrimitive();

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

    size_t rdmaInitSize = 512 * 1024 * 1024;
    rdmaCommPrimitive = new RdmaCommPrimitive();
    rdmaCommPrimitive->initialization(ips, ports, myRank, nRanks, localRank, rdmaInitSize);

    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&rdmaSendBuff, rdmaInitSize * sizeof(float)));
    CUDACHECK(cudaMemset(rdmaSendBuff, 0, rdmaInitSize * sizeof(float)));
    rdmaCommPrimitive->RDMA_Register_GPU_MemRegion(rdmaSendBuff, rdmaInitSize);

    // Cuda Stream initialization
    cudaStreamCreate(&cudaStream);

    // Cuda IPC initialization
    if (localRank == 0)
    {
        sharedPipe = new SharedPipe(pipe_name, 2, std::vector<int>({0, 1})); // to do
        cudaIPCCommPrimitive = new CudaIPCCommPrimitive(*sharedPipe);
    }
    else
    {
        sharedPipe = new SharedPipe(pipe_name, 2);
        cudaIPCCommPrimitive = new CudaIPCCommPrimitive(*sharedPipe);
    }

    CUDACHECK(cudaMalloc(&cudaIpcRevBuff, rdmaInitSize * sizeof(float)));
    CUDACHECK(cudaMemset(cudaIpcRevBuff, 0, rdmaInitSize * sizeof(float)));

    // Synchronize
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
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
    cudaStreamDestroy(cudaStream);
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
            bool debugMode = false;
            std::chrono::_V2::system_clock::time_point startTime, endTime;
            if (debugMode)
            {
                startTime = std::chrono::system_clock::now();
            }

            allReduce_Ring(gradients, size, selfName, plan, debugMode);

            if (debugMode)
            {
                endTime = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = endTime - startTime;
                std::cout << "allReduce_Ring Time: " << elapsed_seconds.count() << std::endl;
            }
        }
    }
    else
    {
        std::cout << "Error: " + tensorName + "is not allreduce opreation" << std::endl;
        return 0;
    }

    return 1;
}

int sendReceive(unsigned char **data, int &size, std::string tensorName)
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

void send_MPI(unsigned char **data, int size, std::string selfName, Plan plan)
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

void receive_MPI(unsigned char **data, int &size, std::string selfName, Plan plan)
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

void allReduce_Ring(float *gradients, size_t size, std::string selfName, Plan plan, bool debugMode)
{
    // for time debug
    std::chrono::_V2::system_clock::time_point time1, time2, time3, time4, time5, time6, time7;

    if (debugMode)
    {
        time1 = std::chrono::system_clock::now();
    }

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

    if (debugMode)
    {
        time2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds21 = time2 - time1;
        std::cout << "getEndPoints Time: " << elapsed_seconds21.count() << std::endl;
        time2 = std::chrono::system_clock::now();
    }

    CommunicationType CT = plan.getCommunicationType();
    if (CT == DefaultCT || CT == RdmaCT)
    {
        // CUDACHECK(cudaMemcpy(rdmaSendBuff, gradients, size * sizeof(float), cudaMemcpyDeviceToDevice));
        // cudaDeviceSynchronize();

        cudaMemcpyAsync(rdmaSendBuff, gradients, size * sizeof(float), cudaMemcpyDeviceToDevice, cudaStream);
        cudaStreamSynchronize(cudaStream);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (debugMode)
    {
        time3 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds32 = time3 - time2;
        std::cout << "cudaMemcpy Time: " << elapsed_seconds32.count() << std::endl;
        time3 = std::chrono::system_clock::now();
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
            receiveBuffer = allReduce_Transmit_RDMA(rdmaCommPrimitive, rdmaSendBuff, sendTarget, sendAddress, chunkSize, receiveTarget, receiveAddress, chunkSize, debugMode);
            allReduce_Gradients_SumOrCover(rdmaSendBuff, receiveAddress, receiveBuffer, chunkSize, 0);
        }
        else if (CT == PcieCT)
        {
            receiveBuffer = allReduce_Transmit_CUDAIPC(gradients, sendTarget, sendAddress, chunkSize, receiveTarget, receiveAddress, chunkSize, rank);
            allReduce_Gradients_SumOrCover(gradients, receiveAddress, receiveBuffer, chunkSize, 0);
        }
    }

    if (debugMode)
    {
        time4 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds43 = time4 - time3;
        std::cout << "scatter Time: " << elapsed_seconds43.count() << std::endl;
        time4 = std::chrono::system_clock::now();
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
            receiveBuffer = allReduce_Transmit_RDMA(rdmaCommPrimitive, rdmaSendBuff, sendTarget, sendAddress, chunkSize, receiveTarget, receiveAddress, chunkSize, debugMode);
            allReduce_Gradients_SumOrCover(rdmaSendBuff, receiveAddress, receiveBuffer, chunkSize, 1);
        }
        else if (CT == PcieCT)
        {
            receiveBuffer = allReduce_Transmit_CUDAIPC(gradients, sendTarget, sendAddress, chunkSize, receiveTarget, receiveAddress, chunkSize, rank);
            allReduce_Gradients_SumOrCover(gradients, receiveAddress, receiveBuffer, chunkSize, 1);
        }
    }

    if (debugMode)
    {
        time5 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds54 = time5 - time4;
        std::cout << "allgather Time: " << elapsed_seconds54.count() << std::endl;
        time5 = std::chrono::system_clock::now();
    }


    if (CT == DefaultCT || CT == RdmaCT)
    {
        gradients_Average(rdmaSendBuff, size, endPointsCount, cudaStream);
    }
    else if (CT == PcieCT)
    {
        gradients_Average(gradients, size, endPointsCount, cudaStream);
    }
    cudaStreamSynchronize(cudaStream);

    if (debugMode)
    {
        time6 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds65 = time6 - time5;
        std::cout << "gradients_Average Time: " << elapsed_seconds65.count() << std::endl;
        time6 = std::chrono::system_clock::now();
    }

    if (CT == DefaultCT || CT == RdmaCT)
    {
        // CUDACHECK(cudaMemcpy(gradients, rdmaSendBuff, size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaMemcpyAsync(gradients, rdmaSendBuff, size * sizeof(float), cudaMemcpyDeviceToDevice, cudaStream);
        cudaStreamSynchronize(cudaStream);
    }

    if (debugMode)
    {
        time7 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds76 = time7 - time6;
        std::cout << "cudaMemcpy Time: " << elapsed_seconds76.count() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

float *allReduce_Transmit_RDMA(RdmaCommPrimitive *rdmaCommPrimitive, float *gradients, int sendTarget, int sendAddress, int sendLength, int receiveTarget, int receiveAddress, int receiveLength, bool debugMode)
{
    // for time debug
    std::chrono::_V2::system_clock::time_point startTime, endTime;

    if (debugMode)
    {
        startTime = std::chrono::system_clock::now();
    }

    float *receiveBuffer = rdmaCommPrimitive->send_receive(gradients, sendTarget, sendAddress, sendLength, receiveTarget, receiveAddress, receiveLength);

    if (debugMode)
    {
        endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = endTime - startTime;
        std::cout << "*******send_receive Time: " << elapsed_seconds.count() << std::endl;
    }

    return receiveBuffer;
}

void allReduce_Gradients_SumOrCover(float *gradients, int receiveAddress, float *receiveBuffer, int receiveLength, int type)
{
    float *gradientsOffset = gradients + receiveAddress;
    if (type == 0)
    {
        gradients_Reduce(gradientsOffset, receiveBuffer, receiveLength, cudaStream);
    }
    else if (type == 1)
    {
        // cudaMemcpy(gradientsOffset, receiveBuffer, receiveLength * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(gradientsOffset, receiveBuffer, receiveLength * sizeof(float), cudaMemcpyDeviceToDevice, cudaStream);
    }

    // cudaDeviceSynchronize();
    cudaStreamSynchronize(cudaStream);

    MPI_Barrier(MPI_COMM_WORLD);
}

float *allReduce_Transmit_CUDAIPC(float *gradients, int sendTarget, int sendAddress, int sendLength, int receiveTarget, int receiveAddress, int receiveLength, int myRank)
{
    // if (sendTarget == 0) {
    //     sendTarget = 0;
    //     receiveTarget = 1;
    // } else {
    //     sendTarget = 1;
    //     receiveTarget = 0;
    // }


    // std::cout << "=======================allReduce_Transmit_CUDAIPC1==========================" << std::endl;
    // std::cout << sendAddress << " " << sendLength << " " << sendTarget << std::endl;


    cudaIPCCommPrimitive->run_write_device(gradients + sendAddress, sendLength, 0, 2, sendTarget);
    // std::cout << "=======================allReduce_Transmit_CUDAIPC2==========================" << std::endl;
    cudaIPCCommPrimitive->run_read_device(cudaIpcRevBuff, receiveLength, 0, 2, myRank);
    // std::cout << "=======================allReduce_Transmit_CUDAIPC3==========================" << std::endl;

    return cudaIpcRevBuff;
}