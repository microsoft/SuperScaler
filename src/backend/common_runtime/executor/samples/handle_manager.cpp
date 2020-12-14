// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @Author: Xuhao Luo
 * This code shows a simple usage of HandleManager. The parent process will create
 * a cudaIpcMemHandle and pass it to the child process (though the ipc ring buffer queue).
 * The child process will use the handle manager to manage the handle-to-address mapping.
 *
 * TODO: This code sample only shows getting address from ipc mem handle. For freeing
 * memory, after the memory is freed in parent address, it needs to inform the child
 * process about that and the child process will call HandleManager::free_address to
 * evict the handle. Try add some ipc method to achieve that functionality.
 */
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/types.h>
#include <wait.h>

#include "../utils/shared_memory.hpp"
#include "../utils/ring_buffer.hpp"
#include "../cuda_ipc/cuda_ipc_internal.hpp"
#include "../cuda_ipc/handle_manager.hpp"

#define MAGIC_NUM 42

constexpr char shm_name[] = "ipcShm";
constexpr size_t test_size = 1024;
constexpr size_t queue_size = 10;
constexpr bool p2p_enable = false;

SharedMemory *parent_process()
{
    checkCudaErrors(cudaSetDevice(0));
    SharedMemory *shm;
    // Setup shared memory
    try {
        shm = new SharedMemory(SharedMemory::OpenType::e_create, shm_name);
        shm->truncate(sizeof(RingBufferQueue<cudaIpcMemHandle_t>) + queue_size * sizeof(cudaIpcMemHandle_t));
    } catch (std::runtime_error &e) {
        std::cout << "[Parent] Create shared memory error: " << e.what() << std::endl;
        return shm;
    }

    void *devPtr;
    cudaIpcMemHandle_t handle;
    RingBufferQueue<cudaIpcMemHandle_t> *handle_queue = new (shm->get_ptr()) RingBufferQueue<cudaIpcMemHandle_t>(queue_size * sizeof(cudaIpcMemHandle_t));
    checkCudaErrors(cudaMalloc(&devPtr, test_size));
    checkCudaErrors(cudaIpcGetMemHandle(&handle, devPtr));
    checkCudaErrors(cudaMemset(devPtr, MAGIC_NUM, test_size));

    // Pass handle to child process
    handle_queue->push(handle);
    std::cout << "[Parent] Finish\n";

    return shm;
}

void child_process()
{
    checkCudaErrors(cudaSetDevice(0));
    HandleManager mgr;
    SharedMemory *shm;
    char *host_buffer = new char[test_size];
    char *addr1, *addr2;
    cudaIpcMemHandle_t handle;
    RingBufferQueue<cudaIpcMemHandle_t> *handle_queue;

    // Setup shared memory
    sleep(1);
    try {
        shm = new SharedMemory(SharedMemory::OpenType::e_open, shm_name);
    } catch (std::runtime_error &e) {
        std::cout << "[Child] Open shared memory error: " << e.what() << std::endl;
        goto clean_up;
    }

    handle_queue = static_cast<RingBufferQueue<cudaIpcMemHandle_t> *>(shm->get_ptr());
    // Get handle from parent address
    while (!handle_queue->pop(handle))
        ;

    // Get address from handle for the first time
    addr1 = (char *)mgr.get_address(handle, 0, 0, p2p_enable);
    // Should get the same address from the same handle
    addr2 = (char *)mgr.get_address(handle, 0, 0, p2p_enable);

    if (addr1 != addr2) {
        std::cout << "[Child] Get address error" << std::endl;
        goto clean_up;
    }

    checkCudaErrors(cudaMemcpy(host_buffer, addr2, test_size, cudaMemcpyDefault));

    for (int i = 0; i < test_size; i++) {
        if (host_buffer[i] != MAGIC_NUM)
            std::cout << "[Child] " << i << "th data error" << std::endl;
    }

    std::cout << "[Child] Finish" << std::endl;

clean_up:
    delete shm;
    delete host_buffer;
}

int main()
{
    pid_t pid = fork();
    int wstatus;
    SharedMemory *shm;

    if (pid == 0) {
        child_process();
    } else {
        shm = parent_process();
        wait(&wstatus);
    }

    // unmap shared mem after child process exits
    delete shm;

    return 0;
}
