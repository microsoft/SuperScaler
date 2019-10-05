#include "rdma.h"

//#define PRINTV(x) std::cout << #x" = " << std::to_string(x) << std::endl;
#define PRINTV(x) {}
const bool verbose = false;

std::mutex m;
std::condition_variable cv;
bool ready = false;

std::atomic<uint32_t> stage;

struct remote_region {
	volatile void *remote_addr;
	volatile unsigned remote_key;
} rmr;

std::vector<wolong::RDMAChannel*> channels;
std::vector<remote_region> local_comm_ranks_;
std::vector<remote_region> gpu_comm_ranks_;
CfgTable RDMA_cfg;

wolong::RDMADeviceManager *rdm;

wolong::RDMADevice *rdma_dev;

const size_t gpu_page_size = 64*1024;
// size_t size_cpu = 64*1024*1024;//(size + gpu_page_size - 1) & ~(gpu_page_size - 1);

struct ibv_mr *sending_lmr;
struct ibv_mr *cpu_lmr;
struct ibv_mr *gpu_lmr;
struct ibv_mr *lmr;
struct ibv_mr *lmr2;

// std::string ip[2] = {"10.0.0.21", "10.0.0.25"};
// int port[2] = {10001,10001};

void memcpy_cb(void *arg, enum ibv_wc_status status) {
	stage++;
}

void rpc_func(void *buff, size_t size, void *arg)
{
	struct remote_region *msg_rmr = (struct remote_region *)buff;
	struct remote_region *rmr = (struct remote_region *)arg;
	rmr->remote_addr = msg_rmr->remote_addr;
	rmr->remote_key = msg_rmr->remote_key;	
}

void set_cfg_RDMA(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients, float *gradients_gpu, float *buf_gpu, size_t size)
{
    //RDMA_cfg = cfg;
    RDMA_cfg.parse_excution_plan("RDMA_configure.cfg");
    auto plan = RDMA_cfg.cfg_table["allreduce.classifier.6.bias"];
    std::vector<std::string> host_ip = plan.host_ip;
    std::vector<std::string> host_port = plan.host_port;

    rdm = new wolong::RDMADeviceManager(2, 2, host_ip[myRank], std::stoi(host_port[myRank]));
	rdm->Init();

    std::cout << host_ip[myRank] << ":" <<std::stoi(host_port[myRank])<<"\n";

    rdma_dev = rdm->GetDevice(0);

    sending_lmr = rdma_dev->AllocateMemRegion(1*1024*1024);
	cpu_lmr = rdma_dev->AllocateMemRegion(size * 4);
    gpu_lmr = rdma_dev->RegisterMemRegion(buf_gpu, size * 4);

    lmr = rdma_dev->RegisterMemRegion(gradients, size * 4);
    lmr2 = rdma_dev->RegisterMemRegion(gradients_gpu, size * 4);
  
    local_comm_ranks_ = std::vector<remote_region>(nRanks);
    local_comm_ranks_[myRank].remote_addr = cpu_lmr->addr;
    local_comm_ranks_[myRank].remote_key  = cpu_lmr->rkey;

    gpu_comm_ranks_ = std::vector<remote_region>(nRanks);
    gpu_comm_ranks_[myRank].remote_addr = gpu_lmr->addr;
    gpu_comm_ranks_[myRank].remote_key  = gpu_lmr->rkey;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), sizeof(remote_region) / 4,
                MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gpu_comm_ranks_.data(), sizeof(remote_region) / 4,
                MPI_INT, MPI_COMM_WORLD);

    channels = std::vector<wolong::RDMAChannel *>(nRanks);
    for(int i = 0; i < nRanks; i++)
    {
        if(i!=myRank)
        channels[i] = rdma_dev->GetRDMAChannelWithIdx(host_ip[i], std::stoi(host_port[i]), 1);
    }

}

void RDMA_scaler_all_reduce_host(float *gradients, size_t size, int myRank, int nRanks, int localRank)
{

    stage = 0;
    int i = 0;
    //fprintf(stderr, "rank = %d cpu_rmr.remote_addr %p, cpu_rmr.remote_key %x\n", myRank, local_comm_ranks_[myRank^1].remote_addr, local_comm_ranks_[myRank^1].remote_key);
    auto plan = RDMA_cfg.cfg_table["allreduce.classifier.6.bias"];

    for(auto op_ :plan.operation)
    {
        i++;
        if(op_.operation_type == "write")
        {
            if(op_.send_target[myRank] != -1)
            channels[op_.send_target[myRank]]->Memcpy(lmr->addr + op_.send_address[myRank] * sizeof(float), lmr, 
                                                    (void *)local_comm_ranks_[op_.send_target[myRank]].remote_addr + op_.send_address[myRank] * sizeof(float), 
                                                    local_comm_ranks_[op_.send_target[myRank]].remote_key, 
                                                    op_.send_length[myRank] * sizeof(float), MEMCPY_LOCAL_TO_REMOTE, memcpy_cb, nullptr);
            while (stage!=i);
            MPI_Barrier(MPI_COMM_WORLD);
            if(op_.receive_target[myRank] != -1)
            {
                float *buf = (float *)cpu_lmr->addr + op_.receive_address[myRank];
                float *grad = gradients + op_.receive_address[myRank];
                if(op_.average)
                    for(int i = 0 ; i < op_.receive_length[myRank]; i++)
                        grad[i] += buf[i];
                else
                    for(int i = 0 ; i < op_.receive_length[myRank]; i++)
                        grad[i] = buf[i];
                
            }
            
        }

    }
    for (int i = 0; i < size; i++)
    {
        gradients[i] /= nRanks;
    }


}

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)
//inline void print_gpu_array(float* array, std::string name = "", size_t size = 10){}
// inline void print_gpu_array(float* array, std::string name = "", size_t size = 10){
//     float *tmp = (float *)malloc(sizeof(float) * size);
//     cudaMemcpy(tmp, array, sizeof(float) * size, cudaMemcpyDeviceToHost);
//     std::cout << "print_gpu_array (" << name << "): ";
//     for (size_t i = 0; i < size; i++)
//         std::cout << ", " << tmp[i];
//     std::cout << std::endl;
//     free(tmp);
// }
inline void print_gpu_array(float* array, std::string name = "", size_t size = 10){
    float *tmp = (float *)malloc(sizeof(float) * size);
    CUDACHECK(cudaMemcpy(tmp, array, sizeof(float) * size, cudaMemcpyDeviceToHost));
    std::cout << "print_gpu_array (" << name << "): ";
    for (size_t i = 0; i < size; i++)
        std::cout << ", " << tmp[i];
    std::cout << std::endl;
    free(tmp);
}

void RDMA_scaler_all_reduce_device(float *gradients, size_t size, int myRank, int nRanks, int localRank)
{
    auto start_time = std::chrono::system_clock::now();

    stage = 0;
    int i = 0;
    //fprintf(stderr, "rank = %d cpu_rmr.remote_addr %p, cpu_rmr.remote_key %x\n", myRank, local_comm_ranks_[myRank^1].remote_addr, local_comm_ranks_[myRank^1].remote_key);
    auto plan = RDMA_cfg.cfg_table["allreduce.classifier.6.bias"];

    for(auto op_ :plan.operation)
    {
        i++;
        if(op_.operation_type == "write")
        {
            if(op_.send_target[myRank] != -1){
                channels[op_.send_target[myRank]]->Memcpy(lmr2->addr + op_.send_address[myRank] * sizeof(float), lmr2,
                                                          (void *)gpu_comm_ranks_[op_.send_target[myRank]].remote_addr + op_.send_address[myRank] * sizeof(float),
                                                          gpu_comm_ranks_[op_.send_target[myRank]].remote_key,
                                                          op_.send_length[myRank] * sizeof(float), MEMCPY_LOCAL_TO_REMOTE, memcpy_cb, nullptr);
                if (verbose)
                {
                    std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                    std::cout << "\t [rdma_Memcpy][" << i << "], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
                }
            }
            while (stage!=i); //wait for memcpy finish
            if (verbose)
            {
                std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                std::cout << "\t [rdma_Memcpy-while][" << i <<"], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (verbose)
            {
                std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                std::cout << "\t [rdma_Memcpy-while-barrier][" << i <<"], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
            }
            if (op_.receive_target[myRank] != -1)
            {
                float *buf = (float *)gpu_lmr->addr + op_.receive_address[myRank];
                float *grad = gradients + op_.receive_address[myRank];
                if (op_.average)
                {
                    gradients_Reduce(grad, buf, op_.receive_length[myRank]);
                    cudaDeviceSynchronize(); //TODO check necessary w.r.t. loop
                    if (verbose)
                    {
                        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                        std::cout << "\t [rdma_Memcpy-barrier-while-reduce][" << i << "], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
                    }
                }
                else
                {
                    cudaMemcpy(grad, buf, op_.receive_length[myRank] * sizeof(float), cudaMemcpyDeviceToDevice);
                    //cudaDeviceSynchronize();
                    if (verbose)
                    {
                        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                        std::cout << "\t [rdma_Memcpy-barrier-while-cudamemcpy][" << i << "], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
                    }
                }
            }
        }
    }
    gradients_Average(gradients, size, nRanks);
    cudaDeviceSynchronize();
    if (verbose)
    {
        std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
        std::cout << "\t [rdma_Memcpy-gradients.Avg], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
    }
}
/*
void RDMA_scaler_all_reduce_host(float *gradients, int size, int myRank, int nRanks, int localRank)
{
    wolong::RDMADeviceManager *rdm = new wolong::RDMADeviceManager(2, 2, ip[myRank], port[myRank]);
	rdm->Init();

    wolong::RDMADevice *rdma_dev = rdm->GetDevice(0);
    
    const size_t gpu_page_size = 64*1024;
    size_t size_cpu = (size + gpu_page_size - 1) & ~(gpu_page_size - 1);

    struct ibv_mr *sending_lmr = rdma_dev->AllocateMemRegion(1*1024*1024);
	struct ibv_mr *cpu_lmr = rdma_dev->AllocateMemRegion(size_cpu * 4);
    struct ibv_mr *lmr = rdma_dev->RegisterMemRegion(gradients, size * 4);

	fprintf(stderr, "rank = %d cpu_lmr->addr %p, cpu_lmr->rkey %x\n", myRank, cpu_lmr->addr, cpu_lmr->rkey);

    local_comm_ranks_ = std::vector<remote_region>(nRanks);
    local_comm_ranks_[myRank].remote_addr = cpu_lmr->addr;
    local_comm_ranks_[myRank].remote_key  = cpu_lmr->rkey;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), sizeof(remote_region) / 4,
                MPI_INT, MPI_COMM_WORLD);

	wolong::RDMAChannel *channel = rdma_dev->GetRDMAChannelWithIdx(ip[myRank^1], port[myRank^1], 1);

    rmr.remote_addr = nullptr;


    auto start_time = std::chrono::system_clock::now();
    fprintf(stderr, "rank = %d cpu_rmr.remote_addr %p, cpu_rmr.remote_key %x\n", myRank, local_comm_ranks_[myRank^1].remote_addr, local_comm_ranks_[myRank^1].remote_key);

    channel->Memcpy(lmr->addr, lmr, (void *)local_comm_ranks_[myRank^1].remote_addr, local_comm_ranks_[myRank^1].remote_key, size*4, MEMCPY_LOCAL_TO_REMOTE, memcpy_cb, nullptr);

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), sizeof(remote_region) / 4,
                  MPI_INT, MPI_COMM_WORLD);

    float *buf = (float *)cpu_lmr->addr;
    for (int i = 0; i < size; i++)
    {
        if(gradients[i] !=  buf[i])
        fprintf(stderr, "rank = %d, gradients[%d] = %f, receive[%d] = %f\n",myRank,i,gradients[i],i,buf[i]);
        gradients[i] = (gradients[i] + buf[i]) / nRanks;
    }
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start_time;
	std::cout << "------------elapsed time: " << elapsed_seconds.count() <<'\n';
}
*/