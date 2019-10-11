#include <string>
#include <iostream>
#include <cuda_runtime.h>

inline void print_cpu_array(float* array, std::string name = "", size_t size = 10){
    std::cout << "print_cpu_array (" << name << "): ";
    for (size_t i = 0; i < size; i++)
        std::cout << ", " << array[i];
    std::cout << std::endl;
}

inline void print_gpu_array(float* array, std::string name = "", size_t size = 10){
    float *tmp = (float *)malloc(sizeof(float) * size);
    CUDACHECK(cudaMemcpy(tmp, array, sizeof(float) * size, cudaMemcpyDeviceToHost));
    std::cout << "print_gpu_array (" << name << "): ";
    for (size_t i = 0; i < size; i++)
        std::cout << ", " << tmp[i];
    std::cout << std::endl;
    free(tmp);
}