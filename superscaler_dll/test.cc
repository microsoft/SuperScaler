#include "super_scaler.h"

int main()
{
    int myRank = 0, nRanks = 0, localRank = 0;
    PlanTable table;
    initialization(myRank, nRanks, localRank, table);

    // prepare tensor data (CUDA memery)
    std::string tensorName = "For_gradients/conv1/conv2d/Conv2D_grad/tuple/control_dependency_1";
    size_t size = 16 * 1024 * 1024;
    float *gradients = new float[size];
    for (int i = 0; i < size; i++)
    {
        gradients[i] = (myRank + 1) * 2 * i;
    }
    float *sendbuff = nullptr;
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff, gradients, size * sizeof(float), cudaMemcpyHostToDevice));

    int testTimes = 10;
    double testSecond = 0;
    while (testTimes--)
    {
        // all reduce
        std::cout << "Before allReduce: ";
        for (int i = 0; i < size; i++)
        {
            gradients[i] = 0;
        }
        CUDACHECK(cudaMemcpy(gradients, sendbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 16; i++)
        {
            std::cout << gradients[i] << " ";
        }
        std::cout << std::endl;

        auto startTime = std::chrono::system_clock::now();
        int ret = allReduce(sendbuff, size, tensorName, table);
        auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = endTime - startTime;
        testSecond += elapsed_seconds.count();

        if (ret == 0)
        {
            std::cout << "AllReduce Error" << std::endl;
            return 0;
        }

        std::cout << "After allReduce: ";
        for (int i = 0; i < size; i++)
        {
            gradients[i] = 0;
        }
        CUDACHECK(cudaMemcpy(gradients, sendbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 16; i++)
        {
            std::cout << gradients[i] << " ";
        }
        std::cout << std::endl;
    }
    testSecond /= 10;
    std::cout << "test_device_rdma, gradient size: " << std::to_string(size) << ", elapsed time: " << testSecond << "s, Throughput: " << std::to_string(size * 4 / testSecond / 1024 / 1024 / 1024) << "GB/s\n";

    // release memery
    CUDACHECK(cudaFree(sendbuff));
    delete[] gradients;

    finalization();

    std::cout << "All Done" << std::endl;
    return 0;
}
// make clean
// make
// /usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.21 ./test_atomic_operations
// /usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.21 ./test
// /usr/bin/mpirun -n 2 -H 10.0.0.21,10.0.0.25 -bind-to none -map-by slot -x CUDA_VISIBLE_DEVICES=2 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib ./test
