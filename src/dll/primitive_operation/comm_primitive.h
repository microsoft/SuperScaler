#include <string>
#include <iostream>
#include <mpi.h>
#include "../config_parse/parse.h"
#include "../rdma/rdma.h"

							 


struct remote_region {
	volatile void *remote_addr;
	volatile unsigned remote_key;
};


class CommPrimitive{
public:
    CommPrimitive() {}
    ~CommPrimitive() {}
    virtual void run_send_recieve_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_send_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_recieve_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_read_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_write_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_send_recieve_device(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_send_device(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_recieve_device(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_read_device(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void run_write_device(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    virtual void execute(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_) {}
    void getinfo(){
        std::cout << "Communication Lib type: " << lib_type << std::endl;
    }
    CfgTable get_rdma_cfg(){
        return this->RDMA_cfg;
    }

protected:
    std::string lib_type;
<<<<<<< HEAD
    CfgTable RDMA_cfg;

=======
    std::atomic<uint32_t> stage_;
>>>>>>> 4c30ce038838ab9b78b7c8c19cc6962492963583
};

class RdmaCommPrimitive : public CommPrimitive{
public:
    RdmaCommPrimitive(){
        lib_type = "rdma";
        stage_ = 0;
    }
    ~RdmaCommPrimitive() {}
<<<<<<< HEAD
    void set_cfg_RDMA_host(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients, size_t size);
    void set_cfg_RDMA_device(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients_gpu, float *buf_gpu, size_t size);
    void run_write_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);

    void run_write_device(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_);
=======
    void set_cfg_RDMA(CfgTable cfg, int myRank, int nRanks, int localRank, size_t size);
    void RDMA_Register_CPU_MemRegion(float *gradients, size_t size);
    void RDMA_Register_GPU_MemRegion(float *gradients, size_t size);
    void run_write_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);
    void run_write_device(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);
    void execute(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);

    CfgTable get_rdma_cfg()
    {
        return RDMA_cfg;
    }
>>>>>>> 4c30ce038838ab9b78b7c8c19cc6962492963583

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
    uint32_t* sending_buf;

    struct ibv_mr *lmr_cpu;
    struct ibv_mr *lmr_gpu;
						

    std::vector<struct ibv_mr *> cpu_lmr;
    std::vector<struct ibv_mr *> gpu_lmr;

    int count;

    int myRank_;
    int nRanks_;

    float **buf_gpu = (float **)malloc(sizeof(float *));

};

class MpiCommPrimitive : public CommPrimitive{
public:
    MpiCommPrimitive(){
        lib_type = "mpi";
		buffer = new float[64*1024*1024];
    }
    ~MpiCommPrimitive() {}
    void run_send_recieve_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);

    void run_send_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);

    void run_recieve_host(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);

    void execute(float *gradients, int size, 
                 int myRank, int nRanks, int localRank, excution_operation op_);

protected:
    float * buffer;
};

class CommPrimitiveFactory{
public:
    CommPrimitiveFactory();
    ~CommPrimitiveFactory();
    CommPrimitive* Create(std::string lib_type){
        if(lib_type == "rdma"){
            return new RdmaCommPrimitive();
        } else if(lib_type == "mpi"){
            return new MpiCommPrimitive();
        }
        return new CommPrimitive();
    }
};