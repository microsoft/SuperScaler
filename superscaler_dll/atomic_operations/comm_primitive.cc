#include "comm_primitive.h"

void cb(void *arg, enum ibv_wc_status status)
{
    (*((uint32_t *)arg))++;
}

void RdmaCommPrimitive::execute(float *gradients, int size,
                                int myRank, int nRanks, int localRank, excution_operation op_)
{
    if (op_.operation_type == "write")
    {
        run_write_host(gradients, size,
                       myRank, nRanks, localRank, op_);
    }
    else
    {
        return;
    }
}

void RdmaCommPrimitive::initialization(std::vector<std::string> ips, std::vector<std::string> ports, int myRank, int nRanks, int localRank, size_t size)
{
    count = 0;
    myRank_ = myRank;
    nRanks_ = nRanks;

    rdm = new wolong::RDMADeviceManager(2, 2, ips[myRank], std::stoi(ports[myRank]));

    std::cout << ips[myRank] << ":" << std::stoi(ports[myRank]) << "\n";

    rdm->Init();
    rdma_dev = rdm->GetDevice(0);

    sending_lmr = rdma_dev->AllocateMemRegion(size);
    cpu_lmr = std::vector<struct ibv_mr *>(nRanks * nRanks);
    gpu_lmr = std::vector<struct ibv_mr *>(nRanks * nRanks);
    local_comm_ranks_ = std::vector<remote_region>(nRanks * nRanks);
    gpu_comm_ranks_ = std::vector<remote_region>(nRanks * nRanks);

    sending_buf = (uint32_t *)sending_lmr->addr;

    for (int i = 0; i < nRanks; i++)
    {
        cudaSetDevice(localRank * 1 + 0); //TODO
        cudaMalloc(buf_gpu + i, size * sizeof(float));
        cudaMemset(buf_gpu[i], 0, size * sizeof(float));
        if (i != myRank)
        {
            cpu_lmr[myRank * nRanks + i] = rdma_dev->AllocateMemRegion(size * 4);
            gpu_lmr[myRank * nRanks + i] = rdma_dev->RegisterMemRegion(buf_gpu[i], size * 4);

            local_comm_ranks_[myRank * nRanks + i].remote_addr = cpu_lmr[myRank * nRanks + i]->addr;
            local_comm_ranks_[myRank * nRanks + i].remote_key = cpu_lmr[myRank * nRanks + i]->rkey;

            gpu_comm_ranks_[myRank * nRanks + i].remote_addr = gpu_lmr[myRank * nRanks + i]->addr;
            gpu_comm_ranks_[myRank * nRanks + i].remote_key = gpu_lmr[myRank * nRanks + i]->rkey;
        }
    }

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), sizeof(remote_region) / 4 * nRanks,
                  MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gpu_comm_ranks_.data(), sizeof(remote_region) / 4 * nRanks,
                  MPI_INT, MPI_COMM_WORLD);

    channels = std::vector<wolong::RDMAChannel *>(nRanks);
    for (int i = 0; i < nRanks; i++)
    {
        if (i != myRank)
            channels[i] = rdma_dev->GetRDMAChannelWithIdx(ips[i], std::stoi(ports[i]), 1);
    }
}

float *RdmaCommPrimitive::send_receive(float *gradients, int sendTarget, int sendAddress, int sendLength, int receiveTarget, int receiveAddress, int receiveLength)
{
    channels[sendTarget]->Memcpy(
        (void *)((uint8_t *)lmr_gpu->addr + sendAddress * sizeof(float)),                                               // local_addr
        lmr_gpu,                                                                                                        // local_region
        (void *)((uint8_t *)gpu_comm_ranks_[sendTarget * nRanks_ + myRank_].remote_addr + sendAddress * sizeof(float)), // remote_addr
        gpu_comm_ranks_[sendTarget * nRanks_ + myRank_].remote_key,                                                     // remote_key
        sendLength * sizeof(float),                                                                                     // size
        MEMCPY_LOCAL_TO_REMOTE,                                                                                         // direction
        cb,                                                                                                             // cb
        &stage_                                                                                                         // arg
    );

    while (!stage_)
        ;
    stage_ = 0;

    // cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    float *buf = (float *)gpu_lmr[myRank_ * nRanks_ + receiveTarget]->addr + receiveAddress;

    // float *grad = gradients + receiveAddress;
    // if (reduceType == 0)
    // {
    //     gradients_Reduce(grad, buf, receiveLength);
    // }
    // else if (reduceType == 1)
    // {
    //     cudaMemcpy(grad, buf, receiveLength * sizeof(float), cudaMemcpyDeviceToDevice);
    // }

    return buf;
}

void RdmaCommPrimitive::set_cfg_RDMA(int myRank, int nRanks, int localRank, size_t size)
{
    stage_ = 0;
    count = 0;

    std::cout << "set config for host communication based on rdma" << std::endl;

    myRank_ = myRank;
    nRanks_ = nRanks;

    RDMA_cfg.parse_excution_plan("config/RDMA_configure.cfg");
    auto plan = RDMA_cfg.cfg_table["allreduce.classifier.6.bias"];
    std::vector<std::string> host_ip = plan.host_ip;
    std::vector<std::string> host_port = plan.host_port;

    rdm = new wolong::RDMADeviceManager(2, 2, host_ip[myRank], std::stoi(host_port[myRank]));
    rdm->Init();

    std::cout << host_ip[myRank] << ":" << std::stoi(host_port[myRank]) << "\n";

    rdma_dev = rdm->GetDevice(0);

    sending_lmr = rdma_dev->AllocateMemRegion(1 * 1024 * 1024);
    cpu_lmr = std::vector<struct ibv_mr *>(nRanks * nRanks);
    gpu_lmr = std::vector<struct ibv_mr *>(nRanks * nRanks);
    local_comm_ranks_ = std::vector<remote_region>(nRanks * nRanks);
    gpu_comm_ranks_ = std::vector<remote_region>(nRanks * nRanks);

    sending_buf = (uint32_t *)sending_lmr->addr;

    for (int i = 0; i < nRanks; i++)
    {
        cudaSetDevice(localRank * 1 + 0); //TODO
        cudaMalloc(buf_gpu + i, size * sizeof(float));
        cudaMemset(buf_gpu[i], 0, size * sizeof(float));
        if (i != myRank)
        {
            cpu_lmr[myRank * nRanks + i] = rdma_dev->AllocateMemRegion(size * 4);
            gpu_lmr[myRank * nRanks + i] = rdma_dev->RegisterMemRegion(buf_gpu[i], size * 4);

            local_comm_ranks_[myRank * nRanks + i].remote_addr = cpu_lmr[myRank * nRanks + i]->addr;
            local_comm_ranks_[myRank * nRanks + i].remote_key = cpu_lmr[myRank * nRanks + i]->rkey;

            gpu_comm_ranks_[myRank * nRanks + i].remote_addr = gpu_lmr[myRank * nRanks + i]->addr;
            gpu_comm_ranks_[myRank * nRanks + i].remote_key = gpu_lmr[myRank * nRanks + i]->rkey;
        }
    }

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), sizeof(remote_region) / 4 * nRanks,
                  MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gpu_comm_ranks_.data(), sizeof(remote_region) / 4 * nRanks,
                  MPI_INT, MPI_COMM_WORLD);

    channels = std::vector<wolong::RDMAChannel *>(nRanks);
    for (int i = 0; i < nRanks; i++)
    {
        if (i != myRank)
            channels[i] = rdma_dev->GetRDMAChannelWithIdx(host_ip[i], std::stoi(host_port[i]), 1);
    }
}

void RdmaCommPrimitive::RDMA_Register_CPU_MemRegion(float *gradients, size_t size)
{
    lmr_cpu = rdma_dev->RegisterMemRegion(gradients, size * sizeof(float));
}

void RdmaCommPrimitive::RDMA_Register_GPU_MemRegion(float *gradients, size_t size)
{
    lmr_gpu = rdma_dev->RegisterMemRegion(gradients, size * sizeof(float));
}

void RdmaCommPrimitive::run_write_host(float *gradients, int size,
                                       int myRank, int nRanks, int localRank, excution_operation op_)
{
    if (op_.send_target[myRank] != -1)
    {
        channels[op_.send_target[myRank]]->Memcpy((void *)((uint8_t *)lmr_cpu->addr + op_.send_address[myRank] * sizeof(float)), lmr_cpu,
                                                  (void *)((uint8_t *)local_comm_ranks_[op_.send_target[myRank] * nRanks + myRank].remote_addr + op_.send_address[myRank] * sizeof(float)),
                                                  local_comm_ranks_[op_.send_target[myRank] * nRanks + myRank].remote_key,
                                                  op_.send_length[myRank] * sizeof(float), MEMCPY_LOCAL_TO_REMOTE, cb, &stage_);
        while (!stage_)
            ;
        stage_ = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (op_.receive_target[myRank] != -1)
    {
        float *buf = (float *)cpu_lmr[myRank * nRanks + op_.receive_target[myRank]]->addr + op_.receive_address[myRank];
        float *grad = gradients + op_.receive_address[myRank];
        if (op_.reduce_type[myRank] == 0)
        {
            for (int i = 0; i < op_.receive_length[myRank]; i++)
            {
                grad[i] += buf[i];
            }
        }
        else if (op_.reduce_type[myRank] == 1)
        {
            for (int i = 0; i < op_.receive_length[myRank]; i++)
            {
                grad[i] = buf[i];
            }
        }
    }
}

void RdmaCommPrimitive::run_write_device(float *gradients, int size,
                                         int myRank, int nRanks, int localRank, excution_operation op_)
{
    if (op_.send_target[myRank] != -1)
    {
        channels[op_.send_target[myRank]]->Memcpy((void *)((uint8_t *)lmr_gpu->addr + op_.send_address[myRank] * sizeof(float)), lmr_gpu,
                                                  (void *)((uint8_t *)gpu_comm_ranks_[op_.send_target[myRank] * nRanks + myRank].remote_addr + op_.send_address[myRank] * sizeof(float)),
                                                  gpu_comm_ranks_[op_.send_target[myRank] * nRanks + myRank].remote_key,
                                                  op_.send_length[myRank] * sizeof(float), MEMCPY_LOCAL_TO_REMOTE, cb, &stage_);
        while (!stage_)
            ;
        stage_ = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (op_.receive_target[myRank] != -1)
    {
        float *buf = (float *)gpu_lmr[myRank * nRanks + op_.receive_target[myRank]]->addr + op_.receive_address[myRank];
        float *grad = gradients + op_.receive_address[myRank];
        if (op_.reduce_type[myRank] == 0)
        {
            gradients_Reduce(grad, buf, op_.receive_length[myRank]);
        }
        else if (op_.reduce_type[myRank] == 1)
        {
            cudaMemcpy(grad, buf, op_.receive_length[myRank] * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
}

void MpiCommPrimitive::execute(float *gradients, int size,
                               int myRank, int nRanks, int localRank, excution_operation op_)
{
    if (op_.operation_type == "send_receive")
    {
        run_send_recieve_host(gradients, size,
                              myRank, nRanks, localRank, op_);
    }
    else if (op_.operation_type == "send")
    {
        run_send_host(gradients, size,
                      myRank, nRanks, localRank, op_);
    }
    else if (op_.operation_type == "receive")
    {
        run_recieve_host(gradients, size,
                         myRank, nRanks, localRank, op_);
    }
    else
    {
        return;
    }
}

void MpiCommPrimitive::run_send_recieve_host(float *gradients, int size,
                                             int myRank, int nRanks, int localRank, excution_operation op_)
{
    MPI_Status recv_status;
    MPI_Request recv_req;

    if (op_.average)
    {
        float *segment_send = (float *)gradients + op_.send_address[myRank];
        float *segment_receive = (float *)gradients + op_.receive_address[myRank];
        float *segment_buffer = (float *)buffer + op_.receive_address[myRank];

        MPI_Irecv(segment_buffer, op_.receive_length[myRank], MPI_FLOAT,
                  op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_req);
        MPI_Send(segment_send, op_.send_length[myRank], MPI_FLOAT,
                 op_.send_target[myRank], 0, MPI_COMM_WORLD);

        MPI_Wait(&recv_req, &recv_status);

        for (int i = 0; i < op_.receive_length[myRank]; i++)
            segment_receive[i] += segment_buffer[i];
    }
    else
    {
        float *segment_send = (float *)gradients + op_.send_address[myRank];
        float *segment_receive = (float *)gradients + op_.receive_address[myRank];

        MPI_Sendrecv(segment_send, op_.send_length[myRank], MPI_FLOAT,
                     op_.send_target[myRank], 0,
                     segment_receive, op_.receive_length[myRank], MPI_FLOAT,
                     op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_status);
    }
}

void MpiCommPrimitive::run_send_host(float *gradients, int size,
                                     int myRank, int nRanks, int localRank, excution_operation op_)
{
    if (op_.send_target[myRank] == -1)
        return;
    else
    {
        float *segment_send = (float *)gradients + op_.send_address[myRank];
        MPI_Send(segment_send, op_.send_length[myRank], MPI_FLOAT,
                 op_.send_target[myRank], 0, MPI_COMM_WORLD);
    }
}

void MpiCommPrimitive::send(unsigned char **data, int sendTarget, int sendAddress, int sendLength)
{
    MPI_Send(*data + sendAddress, sendLength, MPI_UNSIGNED_CHAR, sendTarget, 0, MPI_COMM_WORLD);
}

void MpiCommPrimitive::recieve(unsigned char **data, int receiveTarget, int receiveAddress, int &receiveLength)
{
    MPI_Status recv_status;
    MPI_Probe(receiveTarget, 0, MPI_COMM_WORLD, &recv_status);
    MPI_Get_count(&recv_status, MPI_UNSIGNED_CHAR, &receiveLength);
    
    *data = new unsigned char[receiveLength];
    MPI_Recv(*data + receiveAddress, receiveLength, MPI_UNSIGNED_CHAR, receiveTarget, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void MpiCommPrimitive::run_recieve_host(float *gradients, int size,
                                        int myRank, int nRanks, int localRank, excution_operation op_)
{
    MPI_Status recv_status;
    MPI_Request recv_req;

    if (op_.receive_target[myRank] == -1)
        return;
    else
    {
        if (op_.average)
        {
            float *segment_receive = (float *)gradients + op_.receive_address[myRank];
            float *segment_buffer = buffer + op_.receive_address[myRank];
            MPI_Irecv(segment_buffer, op_.receive_length[myRank], MPI_FLOAT,
                      op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_req);
            MPI_Wait(&recv_req, &recv_status);
            for (int i = 0; i < op_.receive_length[myRank]; i++)
                segment_receive[i] += segment_buffer[i];
        }
        else
        {
            float *segment_receive = (float *)gradients + op_.receive_address[myRank];
            MPI_Recv(segment_receive, op_.receive_length[myRank], MPI_FLOAT,
                     op_.receive_target[myRank], 0, MPI_COMM_WORLD, &recv_status);
        }
    }
}