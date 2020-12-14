// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <sys/file.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>


#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    constexpr uint64_t buffer_len = 1024;
    char buffer[buffer_len] = {0};
    snprintf(buffer, buffer_len,
             "checkCudaErrors() Runtime API error = %04d \"%s\" from file <%s>, "
             "line %i.\n",
             err, cudaGetErrorString(err), file, line);
    throw std::runtime_error(std::string(buffer));
  }
}
#endif


#ifndef checkCuErrors
#define checkCuErrors(err) __checkCuErrors(err, __FILE__, __LINE__)
inline void __checkCuErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    constexpr uint64_t buffer_len = 1024;
    char buffer[buffer_len] = {0};
    const char *err_name = nullptr;
    const char *err_str = nullptr;
    const char *invalid_value_str = "[Invalid Value]";
    if (CUDA_ERROR_INVALID_VALUE == cuGetErrorName(err, &err_name)) {
        err_name = invalid_value_str;
    }
    if (CUDA_ERROR_INVALID_VALUE == cuGetErrorString(err, &err_str)) {
        err_str = invalid_value_str;
    }
    snprintf(buffer, buffer_len,
             "checkCuErrors() Driver API error = %04d \"%s: %s\" from file <%s>, "
             "line %i.\n",
             err, err_name, err_str, file, line);
    throw std::runtime_error(std::string(buffer));
  }
}
#endif


#ifndef __FUNCTION_NAME__
    #ifdef __linux__
        #define __FUNCTION_NAME__   __func__
    #else
        #define __FUNCTION_NAME__   __FUNCTION__
    #endif
#endif


template<char ... LOCK_NAME>
class GlobalLock {
public:
    static GlobalLock & get_lock() {
        static GlobalLock lock;
        return lock;
    }
    void lock() {
        if (flock(m_fd, LOCK_EX) < 0) {
            throw std::runtime_error(std::string()
            + "Cannot exclusively lock "
            + get_lock_name()
            + " at "
            + __FUNCTION_NAME__);
        }
    }
    void unlock() {
        flock(m_fd, LOCK_UN);
    }
    const char * get_lock_name() const {
        static constexpr char name[sizeof...(LOCK_NAME) + 1] = {LOCK_NAME...,'\0'};
        return name;
    }

    ~GlobalLock() {
        if (m_fd > 0) {
            unlock();
            close(this->m_fd);
        }
    }

private:
    GlobalLock() {
        m_fd = open((std::string() + "/tmp/." + get_lock_name()).c_str(), O_CREAT | O_RDONLY, S_IRUSR | S_IWUSR);
        if (m_fd < 0) {
            m_fd = open(get_lock_name(), O_CREAT | O_RDONLY, S_IRUSR | S_IWUSR);
            if (m_fd < 0) {
                throw std::runtime_error(
                    std::string()
                    + "Cannot open the lock file "
                    + get_lock_name()
                    + " at "
                    + __FUNCTION_NAME__);
            }
        }
    }
    int m_fd;
};

class DeviceContextGuard {
public:
    DeviceContextGuard() {}
    void guard(int device) {
        checkCudaErrors(cudaGetDevice(&m_device));
        checkCudaErrors(cudaSetDevice(device));
    }
    DeviceContextGuard(int device) {
        checkCudaErrors(cudaGetDevice(&m_device));
        checkCudaErrors(cudaSetDevice(device));
    }
    ~DeviceContextGuard() {
        checkCudaErrors(cudaSetDevice(m_device));
    }
private:
    int m_device;
};
