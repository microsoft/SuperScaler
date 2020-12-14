// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <unordered_map>
#include <functional>
#include <string>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

namespace std {
    template<>
    struct hash<cudaIpcMemHandle_t>
    {
        size_t operator()(const cudaIpcMemHandle_t &handle) const
        {
            return hash<string>()(string(handle.reserved, CUDA_IPC_HANDLE_SIZE));
        }
    };

    template<>
    struct equal_to<cudaIpcMemHandle_t>
    {
        bool operator()(const cudaIpcMemHandle_t &h1, const cudaIpcMemHandle_t &h2) const
        {
            return memcmp(h1.reserved, h2.reserved, CUDA_IPC_HANDLE_SIZE) == 0;
        }
    };
};

class HandleManager {
public:
    HandleManager();
    HandleManager(const HandleManager &) = delete;
    HandleManager &operator=(const HandleManager &) = delete;
    ~HandleManager();

    /**
     * @brief Get the mapped address of the ipc mem handle. Store the handle->address
     * mapping into the cache if the handle is new
     * @param handle Cuda ipc memory handle
     * @param dev_id Cuda device ID
     * @return The mapped address.
     */
    void *get_address(const cudaIpcMemHandle_t &handle, int receiver_dev_id,
                                 int sender_dev_id, bool p2p_enable);

    /**
     * @brief Remove the handle->address mapping from the cache and unmap the ipc
     * mmory address.
     * @param handle Handle of the freed address.
     * @return True if the mapping is in the cache, false else;
     *
     * TODO: When to call this API? Memory is freed in another process, so need some
     * way to let this process know the mapped memory is freed in another process. i.e.,
     * call this API in a "cross process" way.
     */
    bool free_address(const cudaIpcMemHandle_t &handle);

private:
    struct HandleInfo {
        void *dev_ptr;
        int dev_id;
    };

    std::unordered_map<cudaIpcMemHandle_t, HandleInfo> m_handle_cache;
};
