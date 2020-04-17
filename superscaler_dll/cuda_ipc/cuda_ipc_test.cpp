#include <iostream>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>

#include "cuda_ipc_api.hpp"
#include "cuda_ipc_comm_primitive.hpp"

#define SUCCESS 0
#define FAILED 1

// CUDA will lose the context after fork, so create a pure environment for each test
// Refer : https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork
#define CREATE_TEST_CONTEXT                                     \
do {                                                            \
    std::cout << "Start : " << __FUNCTION_NAME__ << std::endl;  \
    if (fork() != 0) {                                          \
        int status;                                             \
        wait(&status);                                          \
        if (status == SUCCESS) {                                \
            std::cout << "SUCCESS\n" << std::endl;              \
            return true;                                        \
        } else {                                                \
            std::cout << "FAILED\n"  << std::endl;              \
            return false;                                       \
        }                                                       \
    }                                                           \
} while(0);

bool test_shared_block_visibility() {
    CREATE_TEST_CONTEXT;
    const int target_value = 0x123;
    cudaIpcMemHandle_t handle;
    int pipe_fds[2];
    int status;

    if (pipe(pipe_fds) == -1) {
        perror(strerror(errno));
    }
    if (fork() != 0) {
        SharedBlock block(sizeof(target_value));
        checkCudaErrors(cudaMemcpy(block.get_buffer(), &target_value, sizeof(target_value), cudaMemcpyDefault));
        handle = block.get_handle();
        int ret = write(pipe_fds[1], &handle, sizeof(handle));
        if (ret != sizeof(handle)) {
            printf("ret : %d\n", ret);
            perror(strerror(errno));
        }
        wait(&status);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        if (status != SUCCESS) {
            exit(FAILED);
        }
        exit(SUCCESS);
    } else {
        size_t handle_size = sizeof(handle);
        close(pipe_fds[1]);
        while(handle_size) {
            int ret = read(pipe_fds[0], ((char *)&handle) + (sizeof(handle) - handle_size), handle_size);
            if (ret == -1) {
                printf("ret : %d\n", ret);
                perror(strerror(errno));
                break;
            }
            handle_size -= ret;
        }
        close(pipe_fds[0]);
        SharedBlock block(handle, sizeof(target_value));
        int receive_value = 0;
        checkCudaErrors(cudaMemcpy(&receive_value, block.get_buffer(), sizeof(target_value), cudaMemcpyDefault));
        if (receive_value == target_value) {
            exit(SUCCESS);
        } else {
            exit(FAILED);
        }
    }
}

bool test_shared_table_copy_in_multiprocess() {
    CREATE_TEST_CONTEXT;
    const char * table_name = "shared_table";
    const int block_size = 1024;
    const char * target_str = "Hello World!";
    char buffer[block_size] = {0};
    const std::vector<int> devices = {0, 1};
    int status;

    if (fork() != 0) {
        {
            SharedTable<> shared_table(table_name, block_size, devices.size(), devices);
            checkCudaErrors(cudaMemcpy(shared_table.get_buffer(0), target_str, strlen(target_str), cudaMemcpyDefault));
            wait(&status);
            checkCudaErrors(cudaMemcpy(buffer, shared_table.get_buffer(1), strlen(target_str), cudaMemcpyDefault));
        }
        if ((status == SUCCESS) && (memcmp(buffer, target_str, strlen(target_str)) == 0)) {
            exit(SUCCESS);
        } else {
            exit(FAILED);
        }
    } else {
        {
            SharedTable<> shared_table(table_name, block_size, devices.size());
            shared_table.add_device(0);
            shared_table.add_device(1);
            while(memcmp(buffer, target_str, strlen(target_str)) != 0) {
                checkCudaErrors(cudaMemcpy(buffer, shared_table.get_buffer(0), strlen(target_str), cudaMemcpyDefault));
                sched_yield();
            }
            checkCudaErrors(cudaMemcpy(shared_table.get_buffer(1), shared_table.get_buffer(0), strlen(target_str), cudaMemcpyDefault));
        }
        exit(SUCCESS);
    }
}

bool test_shared_table_copy_in_singleprocess() {
    CREATE_TEST_CONTEXT;
    const char * table_name = "shared_table";
    const int block_size = 1024;
    const char * target_str = "Hello World!";
    char buffer[block_size] = {0};
    {
        SharedTable<> shared_table1(table_name, block_size);
        SharedTable<> shared_table2(table_name, block_size);

        checkCudaErrors(cudaMemcpy(shared_table1.get_buffer(0), target_str, strlen(target_str), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(shared_table2.get_buffer(1), shared_table1.get_buffer(0), strlen(target_str), cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(buffer, shared_table2.get_buffer(1), strlen(target_str), cudaMemcpyDefault));
    }
    if (memcmp(buffer, target_str, strlen(target_str)) == 0) {
        exit(SUCCESS);
    } else {
        exit(FAILED);
    }
}

bool test_shared_pipe_connectivity(size_t length) {
    CREATE_TEST_CONTEXT;
    const char * pipe_name = "shared_pipe";
    char * target_str = static_cast<char *>(calloc(length + 1, 1));
    char * buffer = static_cast<char *>(calloc(length + 1, 1));
    // Create random supplies
    for (size_t i = 0; i < length; i++) {
        // Randomly pick printable characters
        target_str[i] = rand() % (126 - 32) + 32;
    }

    if (fork() != 0) {
        SharedPipe sp(pipe_name, 2, std::vector<int>({0, 1}));
        bool full = false;
        while(!full) {
            size_t length = 0;
            do {
                int ret = sp.write(target_str + length, strlen(target_str) - length, 0);
                length += ret;
                if (length != strlen(target_str)) {
                    full = true;
                }
            } while(length != strlen(target_str));
        }
        size_t length = 0;
        while(length != strlen(target_str)) {
            length += sp.read(buffer + length, strlen(target_str) - length, 1);
        }
        int status = 0;
        wait(&status);
        if (status != SUCCESS) {
            exit(FAILED);
        }
        if (strcmp(buffer, target_str) != 0) {
            std::cerr << "(1)Expected String : " << target_str << std::endl;
            std::cerr << "(1)Result String   : " << buffer << std::endl;
            exit(FAILED);
        }
    } else {
        SharedPipe sp(pipe_name, 2);
        while(sp.get_capacity() != sp.get_size(0));
        while(sp.get_size(0) != 0) {
            size_t length = 0;
            do {
                int ret = sp.read(buffer + length, strlen(target_str) - length, 0);
                length += ret;
            } while(length != strlen(target_str));
        }
        if (strcmp(buffer, target_str) != 0) {
            std::cerr << "(2)Expected String : " << target_str << std::endl;
            std::cerr << "(2)Result String   : " << buffer << std::endl;
            exit(FAILED);
        }
        size_t length = 0;
        while(length != strlen(target_str)) {
            length += sp.write(buffer + length, strlen(target_str) - length, 1);
        }
        exit(SUCCESS);
    }

    free(target_str);
    free(buffer);
    exit(SUCCESS);
}

bool test_shared_pipe_connectivity() {
    CREATE_TEST_CONTEXT
    dup2(open("/dev/null", O_WRONLY), 1);
    for (int i = 1; i < 16; i++) {
        if (!test_shared_pipe_connectivity((size_t)1 << i)) {
            std::cerr << "Failed in length = " << ((size_t)1 << i) << std::endl;
            exit(FAILED);
        }
    }
    exit(SUCCESS);
}

bool test_ipc_comm_primitive() {
    CREATE_TEST_CONTEXT
    const char * pipe_name = "test_comm_primitive";
    std::vector<float> tensor1(32);
    std::vector<float> tensor2(tensor1.size());
    std::vector<float> tensor3(tensor1.size());
    for (size_t i = 0; i < tensor1.size(); i++) {
        tensor1[i] = rand() % 10;
        tensor2[i] = rand() % 10;
        tensor3[i] = tensor1[i] + tensor2[i];
    }
    if (fork() != 0) {
        SharedPipe sp(pipe_name, 2, std::vector<int>({0, 1}));
        CudaIPCCommPrimitive comm(sp);
        comm.run_write_device(tensor1.data(), tensor1.size(), 0, 2, 0);
        tensor2.clear();
        tensor2.resize(tensor1.size());
        comm.run_read_device(tensor2.data(), tensor2.size(), 0, 2, 1);
        std::vector<float> result(tensor1.size());
        std::transform(tensor1.begin(), tensor1.end(), tensor2.begin(), result.begin(), [](float a, float b) { return a + b; });
        int status;
        wait(&status);
        if (status != SUCCESS) {
            exit(FAILED);
        }
        if (result != tensor3) {
            exit(FAILED);
        }
    } else {
        SharedPipe sp(pipe_name, 2);
        CudaIPCCommPrimitive comm(sp);
        comm.run_write_device(tensor2.data(), tensor2.size(), 0, 2, 1);
        tensor1.clear();
        tensor1.resize(tensor2.size());
        comm.run_read_device(tensor1.data(), tensor1.size(), 0, 2, 0);
        std::vector<float> result(tensor1.size());
        std::transform(tensor1.begin(), tensor1.end(), tensor2.begin(), result.begin(), [](float a, float b) { return a + b; });
        if (result != tensor3) {
            exit(FAILED);
        }
    }
    exit(SUCCESS);
}

int main(int argc, char * argv[]) {

    test_shared_block_visibility();
    test_shared_table_copy_in_multiprocess();
    test_shared_table_copy_in_singleprocess();
    test_shared_pipe_connectivity();
    test_ipc_comm_primitive();

    std::cout << "Finish" << std::endl;

    return 0;
}