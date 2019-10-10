#include <string>
#include <iostream>
#include <mpi.h>
#include "../config_parse/parse.h"
#include "../rdma/rdma.h"

// static std::atomic<uint32_t> stage_;


struct remote_region {
	volatile void *remote_addr;
	volatile unsigned remote_key;
};


class CommPrimitive{
public:
    CommPrimitive() {
        lib_type = "uncertain";
    }
    ~CommPrimitive() {}
    virtual void run_send_recieve_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_send_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_recieve_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_read_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_write_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_send_recieve_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_send_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_recieve_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_read_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    virtual void run_write_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {}
    void getinfo(){
        std::cout<< "Communication Lib type: "<< lib_type << std::endl;
    }
    CfgTable get_rdma_cfg(){
        return this->RDMA_cfg;
    }

protected:
    std::string lib_type;
    CfgTable RDMA_cfg;

};

class RdmaCommPrimitive : public CommPrimitive{
public:
    RdmaCommPrimitive() {
        lib_type = "rdma";
    }
    ~RdmaCommPrimitive() {}
    void set_cfg_RDMA_host(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients, size_t size);
    void set_cfg_RDMA_device(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients_gpu, float *buf_gpu, size_t size);
    void run_write_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);

    void run_write_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);

protected:
    std::vector<wolong::RDMAChannel*> channels;
    std::vector<remote_region> local_comm_ranks_;
    std::vector<remote_region> gpu_comm_ranks_;

    // lock_t lmr_lock;
    // std::vector<std::vector<lock_t>> send_enable;
    // std::vector<std::vector<lock_t>> receive_enable;

    wolong::RDMADeviceManager *rdm;
    wolong::RDMADevice *rdma_dev;

    struct ibv_mr *sending_lmr;
    struct ibv_mr *cpu_lmr;
    struct ibv_mr *gpu_lmr;
    struct ibv_mr *lmr;
    struct ibv_mr *lmr2;
};

class MpiCommPrimitive : public CommPrimitive{
public:
    MpiCommPrimitive(){
        lib_type = "mpi";
        buffer_ptr = malloc(16*1024*1024*sizeof(float)); //TO DO
    }
    ~MpiCommPrimitive() {
        free(buffer_ptr);
    }
    void run_send_recieve_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);

    void run_send_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);

    void run_recieve_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);
protected:
    void* buffer_ptr;
};

class CommPrimitiveFactory{
public:
    CommPrimitiveFactory();
    ~CommPrimitiveFactory();
    CommPrimitive* Create(std::string lib_type){
        if(lib_type == "rdma"){
            return new RdmaCommPrimitive();
        }else if(lib_type == "mpi"){
            return new MpiCommPrimitive();
        }
        return new CommPrimitive();
    }
};