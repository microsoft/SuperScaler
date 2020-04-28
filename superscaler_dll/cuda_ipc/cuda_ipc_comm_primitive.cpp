#include <stdexcept>

#include "cuda_ipc_internal.hpp"
#include "cuda_ipc_comm_primitive.hpp"


CudaIPCCommPrimitive::CudaIPCCommPrimitive(SharedPipe & pipe) : m_pipe(pipe)   {
}

void CudaIPCCommPrimitive::run_write_host(float *gradients, int size, 
    int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif        
    ) {
    write(gradients, sizeof(*gradients) * size, localRank);
}

void CudaIPCCommPrimitive::run_read_host(float *gradients, int size, 
    int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif     
    ) {
    read(gradients, sizeof(*gradients) * size, localRank);
}

void CudaIPCCommPrimitive::run_read_device(float *gradients, int size, 
    int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif      
    ) {
    read(gradients, sizeof(*gradients) * size, localRank);
}

void CudaIPCCommPrimitive::run_write_device(float *gradients, int size, 
    int myRank, int nRanks, int localRank
#ifndef WARNING_AS_ERROR
        , excution_operation op_
#endif       
    ) {
    write(gradients, sizeof(*gradients) * size, localRank);
}

void CudaIPCCommPrimitive::read(void * buffer, size_t size, int device) {
    size_t length = 0;
    while (length < size) {
        length += m_pipe.read(reinterpret_cast<char *>(buffer) + length, size - length, device);
    }
}

void CudaIPCCommPrimitive::write(const void * buffer, size_t size, int device) {
    size_t length = 0;
    while (length < size) {
        length += m_pipe.write(reinterpret_cast<const char *>(buffer) + length, size - length, device);
    }
}

