#include "super_scaler.h"

void copyMemeryD2H_Display(float *host, float *device, size_t size, int displaySize = 16)
{
    for (int i = 0; i < size; i++)
    {
        host[i] = 0;
    }
    CUDACHECK(cudaMemcpy(host, device, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < displaySize; i++)
    {
        std::cout << host[i] << " ";
    }
    std::cout << "......";
    for (int i = size - displaySize; i < size; i++)
    {
        std::cout << host[i] << " ";
    }
    std::cout << std::endl;
}

void allReduceTest(int testTimes, float *gradients, float *sendbuff, size_t size, std::string tensorName, int myRank)
{
    double testSecond = 0;
    for (int testi = 0; testi < testTimes; testi++)
    {
        std::cout << "Rank " << myRank << ": Before allReduce: ";
        copyMemeryD2H_Display(gradients, sendbuff, size);

        auto startTime = std::chrono::system_clock::now();
        int ret = allReduce(sendbuff, size, tensorName);
        auto endTime = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = endTime - startTime;
        testSecond += elapsed_seconds.count();

        if (ret == 0)
        {
            std::cout << "AllReduce Error" << std::endl;
            return;
        }

        std::cout << "Rank " << myRank << ": After allReduce: ";
        copyMemeryD2H_Display(gradients, sendbuff, size);
    }
    testSecond /= testTimes;
    std::cout << "allReduceTest, gradient size: " << std::to_string(size) << ", elapsed time: " << testSecond << "s, Throughput: " << std::to_string(size * 4 / testSecond / 1024 / 1024 / 1024) << "GB/s\n";
}

void sendRecvTest(int testTimes, float *gradients, size_t size, std::string tensorName, int myRank)
{
    // Before
    unsigned char *data = NULL;
    int sendRecvSize = 0;
    if (myRank == 0)
    {
        sendRecvSize = size;
        data = new unsigned char[sendRecvSize];
        for (int i = 0; i < sendRecvSize; i++)
        {
            data[i] = gradients[1] + 'A';
        }
        std::cout << "Before sendRecv: Rank " << myRank << ", Size: " << sendRecvSize << ", data: ";
        for (int i = 0; i < 16; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    else if (myRank == 1)
    {
        std::cout << "Before sendRecv: Rank " << myRank << ", Size: " << sendRecvSize << ", data == NULL " << std::endl;
    }

    // Process
    int ret = sendReceive(&data, sendRecvSize, tensorName);

    // After
    std::cout << "After sendRecv: Rank " << myRank << ", Size: " << sendRecvSize << ", data: ";
    for (int i = 0; i < 16; i++)
    {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main()
{
    int myRank = 0, nRanks = 0, localRank = 0;
    initialization(myRank, nRanks, localRank);

    // prepare tensor data (CUDA memery)
    // std::string tensorName = "allReduceTestTensorname";
    std::string tensorName = "allReduceIPCTestTensorname";
    // std::string tensorName = "sendRecvTestTensorname";
    // std::string tensorName = "read_SuperScaler_SubgraphConvs/SuperScaler_Backward_SubgraphBpConvs/SuperScaler_Backward_SubgraphBpConv1/norm1_gradients/SuperScaler_SubgraphConvs/SuperScaler_Backward_SubgraphBpConvs/SuperScaler_Backward_SubgraphBpConv2/conv2_matmul_grad/ShapeN";
    size_t size = 128 * 1024 * 1024;
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = (myRank + 1) * 2;
    }
    float *sendbuff = nullptr;
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff, gradients, size * sizeof(float), cudaMemcpyHostToDevice));

    // test begin
    int testTimes = 10;
    allReduceTest(testTimes, gradients, sendbuff, size, tensorName, myRank);
    // sendRecvTest(testTimes, gradients, size, tensorName, myRank);

    // release memery
    CUDACHECK(cudaFree(sendbuff));
    delete[] gradients;

    finalization();

    std::cout << "All Done" << std::endl;
    return 0;
}
// make clean
// make
// ftp://scaler:scaler@MSRAGPUM21/files/users/v-guanx/models_test/
// /usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.21 ./test_atomic_operations
// /usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.21 ./test
// scp -r seliang@10.0.0.21:/usr/local/include/superscaler_dll /usr/local/include/
// /usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.25 -bind-to none -map-by slot -x CUDA_VISIBLE_DEVICES=2 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib ./test
