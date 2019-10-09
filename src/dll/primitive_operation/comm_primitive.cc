#include "comm_primitive.h"

std::atomic<uint32_t> stage_;
int count;

void memcpy_cb_(void *arg, enum ibv_wc_status status) {
	    stage_ ++;
}

void RdmaCommPrimitive::set_cfg_RDMA_host(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients, size_t size){
    stage_ = 0;
    count = 0; 
    
    std::cout << "set config for host communication based on rdma" << std::endl;


    RDMA_cfg.parse_excution_plan("config/RDMA_configure.cfg");
    auto plan = RDMA_cfg.cfg_table["allreduce.classifier.6.bias"];
    std::vector<std::string> host_ip = plan.host_ip;
    std::vector<std::string> host_port = plan.host_port;

    rdm = new wolong::RDMADeviceManager(2, 2, host_ip[myRank], std::stoi(host_port[myRank]));
    rdm->Init();

    std::cout << host_ip[myRank] << ":" <<std::stoi(host_port[myRank])<<"\n";

    rdma_dev = rdm->GetDevice(0);

    sending_lmr = rdma_dev->AllocateMemRegion(1*1024*1024);
	cpu_lmr = rdma_dev->AllocateMemRegion(size * 4);
    lmr = rdma_dev->RegisterMemRegion(gradients, size * 4);
  
    local_comm_ranks_ = std::vector<remote_region>(nRanks);
    local_comm_ranks_[myRank].remote_addr = cpu_lmr->addr;
    local_comm_ranks_[myRank].remote_key  = cpu_lmr->rkey;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), sizeof(remote_region) / 4,
            MPI_INT, MPI_COMM_WORLD);

    channels = std::vector<wolong::RDMAChannel *>(nRanks);
    for(int i = 0; i < nRanks; i++){
        if(i!=myRank){
            channels[i] = rdma_dev->GetRDMAChannelWithIdx(host_ip[i], std::stoi(host_port[i]), 1);
        }
    }
}

void RdmaCommPrimitive::set_cfg_RDMA_device(CfgTable cfg, int myRank, int nRanks, int localRank, float *gradients_gpu, float *buf_gpu, size_t size){
    stage_ = 0;
    count = 0;
    
    std::cout << "set config for device communication based on rdma" << std::endl;

    RDMA_cfg.parse_excution_plan("config/RDMA_configure.cfg");
    auto plan = RDMA_cfg.cfg_table["allreduce.classifier.6.bias"];
    std::vector<std::string> host_ip = plan.host_ip;
    std::vector<std::string> host_port = plan.host_port;

    rdm = new wolong::RDMADeviceManager(2, 2, host_ip[myRank], std::stoi(host_port[myRank]));
	rdm->Init();

    std::cout << host_ip[myRank] << ":" <<std::stoi(host_port[myRank])<<"\n";

    rdma_dev = rdm->GetDevice(0);

    sending_lmr = rdma_dev->AllocateMemRegion(1*1024*1024);
    gpu_lmr = rdma_dev->RegisterMemRegion(buf_gpu, size * 4);

    lmr2 = rdma_dev->RegisterMemRegion(gradients_gpu, size * 4);

    gpu_comm_ranks_ = std::vector<remote_region>(nRanks);
    gpu_comm_ranks_[myRank].remote_addr = gpu_lmr->addr;
    gpu_comm_ranks_[myRank].remote_key  = gpu_lmr->rkey;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gpu_comm_ranks_.data(), sizeof(remote_region) / 4,
            MPI_INT, MPI_COMM_WORLD);

    channels = std::vector<wolong::RDMAChannel *>(nRanks);

    for(int i = 0; i < nRanks; i++)
    {
        if(i!=myRank){
            channels[i] = rdma_dev->GetRDMAChannelWithIdx(host_ip[i], std::stoi(host_port[i]), 1);
        }
    }
}

void RdmaCommPrimitive::run_write_host(float *gradients, int size, int myRank,
                 int nRanks, int localRank, excution_operation op_) {
    std::cout << "rdma run write host" << std::endl;

    count ++;
    if(op_.send_target[myRank] != -1)
    channels[op_.send_target[myRank]]->Memcpy(lmr->addr + op_.send_address[myRank] * sizeof(float), lmr, 
                                                (void *)local_comm_ranks_[op_.send_target[myRank]].remote_addr + op_.send_address[myRank] * sizeof(float), 
                                                local_comm_ranks_[op_.send_target[myRank]].remote_key, 
                                                op_.send_length[myRank] * sizeof(float), MEMCPY_LOCAL_TO_REMOTE, memcpy_cb_, nullptr);
    while (stage_ != count);
    MPI_Barrier(MPI_COMM_WORLD);
    if(op_.receive_target[myRank] != -1){
        float *buf = (float *)cpu_lmr->addr + op_.receive_address[myRank];
        float *grad = gradients + op_.receive_address[myRank];
        if(op_.average)
            for(int i = 0 ; i < op_.receive_length[myRank]; i++)
                grad[i] += buf[i];
        else
            for(int i = 0 ; i < op_.receive_length[myRank]; i++)
                grad[i] = buf[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void RdmaCommPrimitive::run_write_device(float *gradients, int size, int myRank,
                int nRanks, int localRank, excution_operation op_) {
    std::cout << "rdma run write device" << std::endl;
 
    count ++;
    if(op_.send_target[myRank] != -1){
            channels[op_.send_target[myRank]]->Memcpy(lmr2->addr + op_.send_address[myRank] * sizeof(float), lmr2,
                                                        (void *)gpu_comm_ranks_[op_.send_target[myRank]].remote_addr + op_.send_address[myRank] * sizeof(float),
                                                        gpu_comm_ranks_[op_.send_target[myRank]].remote_key,
                                                        op_.send_length[myRank] * sizeof(float), MEMCPY_LOCAL_TO_REMOTE, memcpy_cb_, nullptr);
                // if (verbose)
                // {
                //     std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                //     std::cout << "\t [rdma_Memcpy][" << i << "], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
                // }
    }
    while (stage_ != count); //wait for memcpy finish
            // if (verbose)
            // {
            //     std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
            //     std::cout << "\t [rdma_Memcpy-while][" << i <<"], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
            // }
    MPI_Barrier(MPI_COMM_WORLD);
            // if (verbose)
            // {
            //     std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
            //     std::cout << "\t [rdma_Memcpy-while-barrier][" << i <<"], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
            // }
    if (op_.receive_target[myRank] != -1){
        float *buf = (float *)gpu_lmr->addr + op_.receive_address[myRank];
        float *grad = gradients + op_.receive_address[myRank];
        if (op_.average){
            gradients_Reduce(grad, buf, op_.receive_length[myRank]);
            cudaDeviceSynchronize(); //TODO check necessary w.r.t. loop
                    // if (verbose)
                    // {
                    //     std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                    //     std::cout << "\t [rdma_Memcpy-barrier-while-reduce][" << i << "], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
                    // }
        }
        else{
            cudaMemcpy(grad, buf, op_.receive_length[myRank] * sizeof(float), cudaMemcpyDeviceToDevice);
                    //cudaDeviceSynchronize();
                    // if (verbose)
                    // {
                    //     std::chrono::duration<double> elapsed_seconds = (std::chrono::system_clock::now() - start_time);
                    //     std::cout << "\t [rdma_Memcpy-barrier-while-cudamemcpy][" << i << "], elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size * 4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";
                    // }
        }
    }
}

void MpiCommPrimitive::run_send_recieve_host(float *gradients, int size, int myRank,
                int nRanks, int localRank, excution_operation op_) {
    std::cout<< myRank << ":run_send&recieve_host" << std::endl;

    MPI_Status recv_status;
    MPI_Request recv_req;
        
    if(op_.average){
        float* buffer = (float*)buffer_ptr;
        float* segment_send = (float*)gradients + op_.send_address[myRank];
        float* segment_receive = (float*)gradients + op_.receive_address[myRank];
        float* segment_buffer = buffer + op_.receive_address[myRank];
        MPI_Irecv(segment_buffer, op_.receive_length[myRank],
                MPI_FLOAT, 
                op_.receive_target[myRank], 
                0, MPI_COMM_WORLD, &recv_req);
        MPI_Send(segment_send, op_.send_length[myRank],
                MPI_FLOAT, 
                op_.send_target[myRank], 
                0, MPI_COMM_WORLD);

        MPI_Wait(&recv_req, &recv_status);

        for(int i = 0 ; i < op_.receive_length[myRank]; i++){
            segment_receive[i] += segment_buffer[i];
        }
    }
    else{
        float* segment_send = (float*)gradients + op_.send_address[myRank];
        float* segment_receive = (float*)gradients + op_.receive_address[myRank];
        MPI_Sendrecv(segment_send, op_.send_length[myRank],
                    MPI_FLOAT,
                    op_.send_target[myRank], 0,
                    segment_receive, op_.receive_length[myRank],
                    MPI_FLOAT,
                    op_.receive_target[myRank], 
                    0, MPI_COMM_WORLD, &recv_status);
    }
}

void MpiCommPrimitive::run_send_host(float *gradients, int size, int myRank,
                int nRanks, int localRank, excution_operation op_) {
    std::cout<< myRank << "run_send_host" << std::endl;

    if(op_.send_target[myRank] == -1)
        return;
    else{
        float* segment_send = (float*)gradients + op_.send_address[myRank];
        MPI_Send(segment_send, op_.send_length[myRank],
                MPI_FLOAT, 
                op_.send_target[myRank], 0, MPI_COMM_WORLD);
    }
}

void MpiCommPrimitive::run_recieve_host(float *gradients, int size, int myRank,
                int nRanks, int localRank, excution_operation op_) {
    std::cout<< myRank << "run_recieve_host" << std::endl;

    MPI_Status recv_status;
    MPI_Request recv_req;

    if(op_.receive_target[myRank] == -1)
        return;
    else{
        if(op_.average){
            float* buffer = (float*)buffer_ptr;
            float* segment_receive = (float*)gradients + op_.receive_address[myRank];
            float* segment_buffer = buffer + op_.receive_address[myRank];
            MPI_Irecv(segment_buffer, op_.receive_length[myRank],
                    MPI_FLOAT, 
                    op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_req);
            MPI_Wait(&recv_req, &recv_status);
            for(int i = 0 ; i < op_.receive_length[myRank]; i++){
                segment_receive[i] += segment_buffer[i];
            }
        }
        else{
            float* segment_receive = (float*)gradients + op_.receive_address[myRank];
            MPI_Recv(segment_receive, op_.receive_length[myRank],
                    MPI_FLOAT, 
                    op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_status);
        }
    }
}