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
    virtual ~CommPrimitive() {}
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

protected:
    std::string lib_type;
    std::atomic<uint32_t> stage_;
};

class RdmaCommPrimitive : public CommPrimitive{
public:
    RdmaCommPrimitive(){
        lib_type = "rdma";
        stage_ = 0;
    }
    ~RdmaCommPrimitive() {}
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

protected:
    std::vector<wolong::RDMAChannel*> channels;
    std::vector<remote_region> local_comm_ranks_;
    std::vector<remote_region> gpu_comm_ranks_;
    CfgTable RDMA_cfg;

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