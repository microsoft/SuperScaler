#pragma once

#include "shared_pipe.hpp"
#ifndef WARNING_AS_ERROR
#include "../atomic_operations/comm_primitive.h"
#endif

class CudaIPCCommPrimitive 
#ifndef WARNING_AS_ERROR
: protected CommPrimitive 
#endif
{
public:
    CudaIPCCommPrimitive(SharedPipe & pipe);

    virtual void run_write_host(float *gradients, int size, 
        int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif     
        );

    virtual void run_read_host(float *gradients, int size, 
        int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif       
        );

    virtual void run_write_device(float *gradients, int size, 
        int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif 
        );

    virtual void run_read_device(float *gradients, int size, 
        int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif      
        );     
private:

    void read(void * buffer, size_t size, int device);
    void write(const void * buffer, size_t size, int device);

    SharedPipe & m_pipe;
};