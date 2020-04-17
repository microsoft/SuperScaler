#pragma once

#include "shared_block.hpp"
#include "shared_table.hpp"
#include "shared_pipe.hpp"


class Pipe {
public:
    size_t write(const void * data, size_t size);
    size_t read(void * data, size_t size);
};


class Board {
public:
    void set(void * buffer, size_t size);
    size_t upload(const void * data, size_t size);
};

class CommunicationManager {
public:
    void set(void * buffer, size_t size, int rankid);

    size_t upload(const void * data, size_t size, int rankid);
    size_t write(const void * data, size_t size, int rankid);
    size_t read(void * data, size_t size, int rankid);
};
