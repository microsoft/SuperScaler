#include "shared_pipe.hpp"

#ifndef TEST
constexpr size_t PIPE_BUFFER_SIZE = 512 * 1024 * 1024; // 512MB
#else
constexpr size_t PIPE_BUFFER_SIZE = 1 * 1024; // 1KB
#endif

size_t SharedPipe::get_available_size(size_t start, size_t end, size_t buffer_size) {
    if (end >= start) {
        return end - start;
    } else {
        return end + buffer_size - start;
    }
}

size_t SharedPipe::get_rest_size(size_t start, size_t end, size_t buffer_size) {
    return buffer_size - get_available_size(start, end, buffer_size) - 1;
}

SharedPipe::SharedPipe(
    const std::string & unique_name
    , size_t device_count) 
: SharedTable<PipeData>(unique_name, PIPE_BUFFER_SIZE, device_count) {
}

SharedPipe::SharedPipe(
    const std::string & unique_name
    , size_t device_count
    , const std::vector<int> & devices
) 
: SharedTable<PipeData>(unique_name, PIPE_BUFFER_SIZE, device_count, devices) {
}

size_t SharedPipe::write(const void * data, size_t size, int device) {
    PipeData & metadata = get_shared_block_metadata(device);
    size_t start = metadata.m_start;
    size_t end = metadata.m_end;
    size_t rest_size = get_rest_size(start, end, get_block_size());
    if (size > rest_size) {
        size = rest_size;
    }
    size_t written_size = 0;
    size_t advance_end = (end + size) % get_block_size();
    char * base = reinterpret_cast<char *>(get_buffer(device));
    if (base == nullptr) {
        throw std::runtime_error(std::string() 
        + "The Buffer of device (" 
        + std::to_string(device) 
        + ") cannot be fetch at " 
        + __FUNCTION_NAME__);
    }
    char * dst = base + end;
    const char * src = reinterpret_cast<const char *>(data);
    if (advance_end >= end) {
        written_size += transfer(dst, src, advance_end - end, device);
    } else {
        written_size += transfer(dst, src, get_block_size() - end, device);
        written_size += transfer(base, src + get_block_size() - end, advance_end, device);
    }
    metadata.m_end = advance_end;
    return written_size;
}

size_t SharedPipe::read(void * data, size_t size, int device) {
    PipeData & metadata = get_shared_block_metadata(device);
    size_t start = metadata.m_start;
    size_t end = metadata.m_end;
    size_t available_size = get_available_size(start, end, get_block_size());
    if (size > available_size) {
        size = available_size;
    }
    size_t read_size = 0;
    size_t advance_start = (start + size) % get_block_size();
    const char * base = reinterpret_cast<char *>(get_buffer(device));
    if (base == nullptr) {
        throw std::runtime_error(std::string() 
        + "The Buffer of device (" 
        + std::to_string(device) 
        + ") cannot be fetch at " 
        + __FUNCTION_NAME__);
    }
    const char * src = base + start;
    char * dst = reinterpret_cast<char *>(data);
    if (advance_start >= start) {
        read_size += transfer(dst, src, advance_start - start, device);
    } else {
        read_size += transfer(dst, src, get_block_size() - start, device);
        read_size += transfer(dst + get_block_size() - start, base, advance_start, device);
    }
    metadata.m_start = advance_start;
    return read_size;    

}

size_t SharedPipe::get_size(int device) {
    PipeData & metadata = get_shared_block_metadata(device);
    size_t start = metadata.m_start;
    size_t end = metadata.m_end;
    return get_available_size(start, end, get_block_size());
}

size_t SharedPipe::get_capacity() const {
    return get_block_size() - 1;
}

size_t SharedPipe::transfer(void * dst, const void * src, size_t size, int device) {
    if (size == 0)
        return 0;
    DeviceContextGuard guard(device);
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));
    return size;
}
