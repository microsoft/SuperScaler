#pragma once

#include "shared_table.hpp"

constexpr size_t PIPE_BUFFER_SIZE = 512 * 1024 * 1024; // 512MB

struct PipeData {
    volatile size_t m_start;
    volatile size_t m_end;
};

class SharedPipe : public SharedTable<PipeData> {
public:
    SharedPipe(
        const std::string & unique_name
        , size_t device_count
        , size_t pipe_size = PIPE_BUFFER_SIZE
    );
    SharedPipe(
        const std::string & unique_name
        , size_t device_count
        , const std::vector<int> & devices
        , size_t pipe_size = PIPE_BUFFER_SIZE
    );

    size_t write(const void * data, size_t size, int device);
    size_t read(void * data, size_t size, int device);
    size_t get_size(int device);
    size_t get_capacity() const;

private:
    size_t transfer(void * dst, const void * src, size_t size, int device);

    inline static size_t get_available_size(size_t start, size_t end, size_t buffer_size);
    inline static size_t get_rest_size(size_t start, size_t end, size_t buffer_size);
};
