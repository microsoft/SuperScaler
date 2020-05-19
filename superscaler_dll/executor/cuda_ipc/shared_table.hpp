#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <mutex>

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "cuda_ipc_internal.hpp"
#include "shared_block.hpp"

#define SHARED_BUFFER_PREFIX "SharedTable_"

using SharedTableLock = GlobalLock<
    'c','u','d','a'
    ,'_'
    ,'i','p','c'
    ,'_'
    ,'s','h','a','r','e','d'
    ,'_'
    ,'l','o','c','k'
    >;

struct EmptyMetadata {
};

template<typename SharedBlockMetadata = EmptyMetadata>
class SharedTable {
public:

    SharedTable() = delete;
    SharedTable(const SharedTable &) = delete;
    SharedTable & operator=(const SharedTable &) = delete;

    SharedTable(const std::string & unique_name, size_t block_size);
    SharedTable(
        const std::string & unique_name
        , size_t block_size
        , size_t device_count                // the number of devices in this table
    );
    SharedTable(
        const std::string & unique_name
        , size_t block_size
        , size_t device_count                // the number of devices in this table
        , const std::vector<int> & devices   // assigned devices in this table
    );
    virtual ~SharedTable();

    bool add_device(int device, void * buffer = nullptr);
    bool is_device_available(int device) const;
    size_t get_block_size() const;
    void * get_buffer(int device);
    SharedBlockMetadata & get_shared_block_metadata(int device);

private:

    void *                                      m_shared_buffer;
    size_t                                      m_shared_buffer_size;
    int                                         m_shared_buffer_handle;
    std::string                                 m_shared_buffer_name;

    typedef struct SharedTableControllerST {
        volatile size_t         m_block_size;
        struct SharedBlockST{
            SharedBlockMetadata m_metadata;
            cudaIpcMemHandle_t  m_handle;
        }                       m_shared_blocks[];
    } SharedTableController;
    SharedTableController *                     m_controller;

    std::vector<std::unique_ptr<SharedBlock> >  m_shared_blocks;

    // If the current SharedTable is the only one occupier to the shared buffer.
    bool is_exclusively_occupied() const;

    static void clear_up_shared_memory();

};


template<typename SharedBlockMetadata>
SharedTable<SharedBlockMetadata>::SharedTable(const std::string & unique_name, size_t block_size) {
    if (unique_name.empty()) {
        throw std::invalid_argument(std::string() + "Empty unique name is forbidden at " + __FUNCTION_NAME__);
    }
    if (block_size == 0) {
        throw std::invalid_argument(std::string() + "Zero block size is forbidden at " + __FUNCTION_NAME__);
    }
    int gpu_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        throw std::runtime_error(std::string() + "GPU count is zero at " + __FUNCTION_NAME__);
    }

    m_shared_buffer_name = SHARED_BUFFER_PREFIX + unique_name;
    m_shared_buffer_size = sizeof(SharedTableController) + gpu_count * (sizeof(struct SharedTableController::SharedBlockST));
    m_shared_blocks.resize(gpu_count);

    std::lock_guard<SharedTableLock > guard(SharedTableLock::get_lock());
    m_shared_buffer_handle = shm_open(m_shared_buffer_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    bool occupied = is_exclusively_occupied();
    if (occupied) {
        // clear up abandoned shared memories
        clear_up_shared_memory();
        // Current process will initialize shared memory
        if (ftruncate(m_shared_buffer_handle, m_shared_buffer_size) != 0) {
            throw std::runtime_error(std::string() + "Cannot truncate the shared memory at " + __FUNCTION_NAME__ + " because " + strerror(errno));
        }
    }

    m_shared_buffer = mmap(0, m_shared_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_shared_buffer_handle, 0);
    if (m_shared_buffer == nullptr) {
        throw std::runtime_error(std::string() + "Cannot open shared memory at" + __FUNCTION_NAME__);
    }

    if (occupied) {
        bzero(m_shared_buffer, m_shared_buffer_size);
    }

    m_controller = static_cast<SharedTableController *>(m_shared_buffer);
    if (occupied) {
        m_controller->m_block_size = block_size;
    } else {
        if (block_size != m_controller->m_block_size) {
            throw std::runtime_error(std::string() + "The block size is not match with previous setting at " + __FUNCTION_NAME__);
        }
    }
}

template<typename SharedBlockMetadata>
SharedTable<SharedBlockMetadata>::SharedTable(
    const std::string & unique_name
    , size_t block_size
    , size_t device_count                // the number of devices in this table
) : SharedTable(unique_name, block_size, device_count, std::vector<int>()) {
}

template<typename SharedBlockMetadata>
SharedTable<SharedBlockMetadata>::SharedTable(
    const std::string & unique_name
    , size_t block_size
    , size_t device_count                // the number of devices in this table
    , const std::vector<int> & devices   // assigned devices in this table
) : SharedTable(unique_name, block_size) {
    for (auto device : devices) {
        if (!add_device(device)) {
            throw std::invalid_argument(
                std::string() 
                + " Cannot add device " 
                + std::to_string(device) 
                + " to table "
                + unique_name
                + " , maybe the device has been added into this table in other place , at "
                + __FUNCTION_NAME__
            );
        }
    }
    size_t available_device = 0;
    while (available_device < device_count) {
        available_device = 0;
        for (size_t i = 0; i < m_shared_blocks.size(); i++) {
            if (is_device_available(i)) {
                available_device++;
            }
        }
    }
    if (available_device > device_count) {
        throw std::invalid_argument(
            std::string()
            + " Available device count (" + std::to_string(available_device)
            + ") is more than the target device count (" + std::to_string(device_count)
            +") at "
            + __FUNCTION_NAME__
        );
    }
}

template<typename SharedBlockMetadata>
SharedTable<SharedBlockMetadata>::~SharedTable() {
    std::lock_guard<SharedTableLock> guard(SharedTableLock::get_lock());
    if (m_shared_buffer) {
        munmap(m_shared_buffer, m_shared_buffer_size);
    }
    if (m_shared_buffer_handle > 0) {
        close(m_shared_buffer_handle);
    }

    if (!m_shared_buffer_name.empty()) {
        shm_unlink(m_shared_buffer_name.c_str());
    }
}

template<typename SharedBlockMetadata>
bool SharedTable<SharedBlockMetadata>::add_device(int device, void * buffer) {
    if (static_cast<unsigned int>(device) >= m_shared_blocks.size()) {
        // Device number is out of range
        throw std::invalid_argument(std::string() + "The device number is out of range at " + __FUNCTION_NAME__);
    }
    std::lock_guard<SharedTableLock> guard(SharedTableLock::get_lock());
    if (m_shared_blocks[device]) {
        // Has been initialized
        return false;
    }
    int current_device;
    checkCudaErrors(cudaGetDevice(&current_device));

    if (is_device_available(device)) {
        m_shared_blocks[device] = std::move(std::unique_ptr<SharedBlock>(new SharedBlock(m_controller->m_shared_blocks[device].m_handle, get_block_size())));
    } else if (buffer == nullptr) {
        checkCudaErrors(cudaSetDevice(device));
        m_shared_blocks[device] = std::move(std::unique_ptr<SharedBlock>(new SharedBlock(get_block_size())));
        m_controller->m_shared_blocks[device].m_handle = m_shared_blocks[device]->get_handle();
    } else {
        m_shared_blocks[device] = std::move(std::unique_ptr<SharedBlock>(new SharedBlock(buffer, get_block_size())));
        m_controller->m_shared_blocks[device].m_handle = m_shared_blocks[device]->get_handle();
    }

    for (size_t i = 0; i < m_shared_blocks.size(); i++) {
        if (i != static_cast<unsigned int>(device) && m_shared_blocks[i]) {
            checkCudaErrors(cudaSetDevice(device));
            checkCudaErrors(cudaDeviceEnablePeerAccess(i, 0));
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cudaDeviceEnablePeerAccess(device, 0));
        }
    }
    checkCudaErrors(cudaSetDevice(current_device));
    return true;
}

template<typename SharedBlockMetadata>
bool SharedTable<SharedBlockMetadata>::is_device_available(int device) const{
    static const cudaIpcMemHandle_t empty_handle = { 0 };
    return memcmp(&(m_controller->m_shared_blocks[device].m_handle), &empty_handle, sizeof(cudaIpcMemHandle_t)) != 0;
}

template<typename SharedBlockMetadata>
size_t SharedTable<SharedBlockMetadata>::get_block_size() const {
    return m_controller->m_block_size;
}

template<typename SharedBlockMetadata>
void * SharedTable<SharedBlockMetadata>::get_buffer(int device) {
    if (!m_shared_blocks[device]) {
        add_device(device);
    }
    return m_shared_blocks[device]->get_buffer();
}

template<typename SharedBlockMetadata>
SharedBlockMetadata & SharedTable<SharedBlockMetadata>::get_shared_block_metadata(int device) {
    if (!m_shared_blocks[device]) {
        add_device(device);
    }
    return m_controller->m_shared_blocks[device].m_metadata;
}

template<typename SharedBlockMetadata>
bool SharedTable<SharedBlockMetadata>::is_exclusively_occupied() const {
    std::stringstream command;

    command 
        << " lsof 2>/dev/null "
        << " | awk " << ( std::string() + "' /" + m_shared_buffer_name +  "/ {print $2} '" )
        << " | sort --unique "
        << " | wc -l ";
    FILE * fp = popen(command.str().c_str(), "r");
    if (fp == NULL) {
        throw std::runtime_error(std::string() + "Cannot list open files at " + __FUNCTION_NAME__);
    }
    
    char buffer[1024 * 1024] = {0};
    std::stringstream stream_buffer;
    while(fgets(buffer, sizeof(buffer), fp)) {
        stream_buffer << buffer;
    }
    pclose(fp);
    unsigned int occupier_count = std::stoul(stream_buffer.str());

    if ( occupier_count == 0) {
        throw std::runtime_error(m_shared_buffer_name + " was not opened by current process at " + __FUNCTION_NAME__);
    } else if (occupier_count == 1) {
        return true;
    } else {
        return false;
    }
}

template<typename SharedBlockMetadata>
void SharedTable<SharedBlockMetadata>::clear_up_shared_memory() {
    char buffer[1024 * 1024] = {0};

    FILE *fp = popen("bash -c 'find /dev/shm | grep " SHARED_BUFFER_PREFIX " ' 2>/dev/null", "r");
    if (fp == NULL) {
        throw std::runtime_error(std::string() + "Cannot retrieve shared memory files at " + __FUNCTION_NAME__);
    }
    std::stringstream active_shared_memories;
    while(fgets(buffer, sizeof(buffer), fp)) {
        active_shared_memories << buffer;
    }
    pclose(fp);

    fp = popen("bash -c ' lsof | grep " SHARED_BUFFER_PREFIX " ' 2>/dev/null", "r");
    if (fp == NULL) {
        throw std::runtime_error(std::string() + "Cannot list open files at " + __FUNCTION_NAME__);
    }
    std::stringstream stream_buffer;
    while(fgets(buffer, sizeof(buffer), fp)) {
        stream_buffer << buffer;
    }
    pclose(fp);

    std::string active_shared_memory;
    const std::string open_files = stream_buffer.str();
    while(std::getline(active_shared_memories, active_shared_memory, '\n')) {
        if (open_files.find(active_shared_memory) == std::string::npos) {
            std::stringstream command;
            command << "rm -f '" << (active_shared_memory) << "' 2>/dev/null ";
            // Try to remove shared memory that wasn't occupied by any process 
            if (system(command.str().c_str()) != 0) {
                // Ignore rm error because rm will fail if this shared memory was created by other user,
                // throw std::runtime_error(std::string() + " Cannot remove unused shared memory" + __FUNCTION_NAME__);
            }
        }
    }

}
