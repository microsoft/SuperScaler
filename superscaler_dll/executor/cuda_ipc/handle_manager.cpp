#include "handle_manager.hpp"
#include "cuda_ipc_internal.hpp"

HandleManager::HandleManager()
{
}

HandleManager::~HandleManager()
{
    for (auto &pair : m_handle_cache)
        checkCudaErrors(cudaIpcCloseMemHandle(pair.second));
}

void *HandleManager::get_address(const cudaIpcMemHandle_t &handle)
{
    auto itr = m_handle_cache.find(handle);
    if (itr == m_handle_cache.end()) {
        void *buffer;
        checkCudaErrors(cudaIpcOpenMemHandle(&buffer, handle, cudaIpcMemLazyEnablePeerAccess));
        m_handle_cache.emplace(handle, buffer);
        return buffer;
    } else {
        return itr->second;
    }
}

bool HandleManager::free_address(const cudaIpcMemHandle_t &handle)
{
    auto itr = m_handle_cache.find(handle);
    if (itr == m_handle_cache.end())
        return false;
    else {
        checkCudaErrors(cudaIpcCloseMemHandle(itr->second));
        m_handle_cache.erase(itr);
        return true;
    }
}
