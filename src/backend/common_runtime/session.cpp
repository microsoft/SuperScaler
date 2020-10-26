#include "session.hpp"
#include "util.hpp"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace superscaler
{
    void Session::Create(const char* plan_fpath)
    {
        LOG(INFO) << "superscaler session is created from " << std::string(plan_fpath);
        auto plan = util::JsonParser::load_from(plan_fpath);
        Create(plan);
    }

    void Session::Create(util::json j)
    {
        ParsePlan(j);
        checkCudaErrors(cudaSetDevice(local_rank_));
        checkCudaErrors(cudaMalloc(&cuda_recv_buf_, cuda_recv_buf_sze_));
        exec_.reset(new PollExecutor(local_rank_));
        //TODO: add rdma channels support
        cuda_channel_ =
            std::make_shared<CudaChannel>(static_cast<uint>(local_rank_), pcie_targets_);
    }

    void Session::Close()
    {
        //those executor-related instances(especially one with cuda-related calls) needs to be destruct explicitly before cuda_recv_buf_ is freed
        //to avoid cuda driver shutting down error
        exec_.reset();
        cuda_channel_.reset();
        rdma_channel_.reset();
        table_.clear();

        checkCudaErrors(cudaSetDevice(local_rank_));
        if (cuda_recv_buf_)
            checkCudaErrors(cudaFree(cuda_recv_buf_));
        LOG(INFO) << "superscaler session is destoryed";
    }

    template <class DataType>
    void Session::AllReduce(const char* tensor_name, DataType* ioput, size_t size, void* stream)
    {
        if (stream)
            throw std::runtime_error("not supported yet!");
        std::lock_guard<std::mutex> lck{sc_mu_};
#ifndef NDEBUG
        LOG(INFO) << "[rank " << global_rank_ << "/" << local_rank_ << "]: allreduce" << tensor_name
                  << " @ " << ioput << " with size: " << size;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
        auto tasks = table_[tensor_name];
        // #ifndef NDEBUG
        LOG(INFO) << tensor_name << " got " << tasks.size() << " plans to run";
        // #endif
        for (auto& element : tasks)
        {
            std::string target_host_id = element["target_host_id"];
            std::string target_device_id = element["target_device_id"];
            std::string key = element["key"];
            std::string op = element["op"];
            std::string tensor_type = element["tensor_type"];
            assert(is_expected_type(ioput, tensor_type));
            std::string route_type = element["route_type"];
            //TODO: add plan validity assertion
            size_t sze = element["size"];
            size_t offset = element["offset"];

            std::hash<std::string> hasher;
            uint64_t msg_id = hasher(key);

            task_id_t send_task_id;
            task_id_t recv_task_id;
            task_id_t reduce_task_id;

            if (route_type == "PCIE")
            {
                if (op == "Send")
                {
                    uint target_id = std::stoi(target_device_id);
                    LOG(INFO) << "[rank " << global_rank_ << "/" << local_rank_ << "]: send "
                              << msg_id << " @ " << ioput << " with size: " << sze << " to "
                              << target_id;
                    send_task_id = exec_->create_task<SendTask>(exec_.get(),
                                                                nullptr,
                                                                cuda_channel_,
                                                                target_id,
                                                                msg_id,
                                                                ioput + offset,
                                                                sze * sizeof(DataType));
                }
                else if (op == "Recv")
                {
                    uint target_id = std::stoi(target_device_id);
                    LOG(INFO) << "[rank " << global_rank_ << "/" << local_rank_ << "]: recv "
                              << msg_id << " @ " << ioput << " with size: " << sze << " from "
                              << target_id;
                    ;
                    std::string reduction_op = element["reduction"];

                    if (reduction_op == "sum")
                    {
                        recv_task_id = exec_->create_task<RecvTask>(exec_.get(),
                                                                    nullptr,
                                                                    cuda_channel_,
                                                                    target_id,
                                                                    msg_id,
                                                                    cuda_recv_buf_,
                                                                    sze * sizeof(DataType));
                        LOG(INFO) << "[rank " << global_rank_ << "/" << local_rank_ << "]: reduce "
                                  << msg_id << " @ " << ioput + offset << " with size: " << sze;
                        reduce_task_id =
                            exec_->create_task<ReductionTask<DataType, SumKernelGPUImpl>>(
                                exec_.get(),
                                nullptr,
                                (const DataType*)cuda_recv_buf_,
                                ioput + offset,
                                SumKernelGPUImpl(),
                                sze);

                        exec_->add_dependence(reduce_task_id, recv_task_id);
                        exec_->add_task(send_task_id);
                        exec_->add_task(recv_task_id);
                        exec_->add_task(reduce_task_id);
                        if (exec_->wait(send_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Send error\n", local_rank_);
                        }
                        if (exec_->wait(recv_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Recv error\n", local_rank_);
                        }
                        if (exec_->wait(reduce_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Reduce error\n", local_rank_);
                        }
                    }
                    else
                    {
                        recv_task_id = exec_->create_task<RecvTask>(
                            exec_.get(),
                            nullptr,
                            cuda_channel_,
                            target_id,
                            msg_id,
                            MemBlock(ioput, offset * sizeof(DataType), sze * sizeof(DataType)));

                        exec_->add_task(send_task_id);
                        exec_->add_task(recv_task_id);
                        if (exec_->wait(send_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Send error\n", local_rank_);
                        }
                        if (exec_->wait(recv_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Recv error\n", local_rank_);
                        }
                    }
                }
            }
            else
            {
                //TODO: RDMA
            }
        }
#ifndef NDEBUG
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        LOG(INFO) << " with time elapsed: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [µs]";
#endif
    }

    template void Session::AllReduce(const char* tensor_name,
                                     float* ioput,
                                     size_t size,
                                     void* stream = nullptr);
    template void Session::AllReduce(const char* tensor_name,
                                     int* ioput,
                                     size_t size,
                                     void* stream = nullptr);

    void Session::Send(const char* tensor_name, unsigned char* input, size_t size)
    {
        //TODO: add send implementation
        //         std::string tensor(tensor_name);
        //         std::hash<std::string> hasher;
        //         int tag = static_cast<long long>(hasher(tensor)) & 0x7FFFFFFF;
        //         int sendTarget = (global_rank_ + 1) % 2;
        // #ifndef NDEBUG
        //         LOG(INFO) << "[rank " << global_rank_ << "]: send to " << sendTarget
        //                   << " with tag: " << tag;
        // #endif
        //         int ret = MPI_Send(input, size, MPI_UNSIGNED_CHAR, sendTarget, tag, MPI_COMM_WORLD);
    }

    void Session::Recv(const char* tensor_name, unsigned char** output, size_t* size)
    {
        //TODO: add recv implementation
        //         std::string tensor(tensor_name);
        //         std::hash<std::string> hasher;
        //         int tag = static_cast<long long>(hasher(tensor)) & 0x7FFFFFFF;
        //         int receiveTarget = (global_rank_ + 1) % 2;
        // #ifndef NDEBUG
        //         LOG(INFO) << "[rank " << global_rank_ << "]: recv from " << receiveTarget
        //                   << " with tag: " << tag;
        // #endif
        //         MPI_Status recv_status;
        //         MPI_Probe(receiveTarget, tag, MPI_COMM_WORLD, &recv_status);
        //         MPI_Get_count(&recv_status, MPI_UNSIGNED_CHAR, (int*)size);
        //         *output = new unsigned char[*size];
        //         int ret = MPI_Recv(*output,
        //                            *size,
        //                            MPI_UNSIGNED_CHAR,
        //                            receiveTarget,
        //                            tag,
        //                            MPI_COMM_WORLD,
        //                            MPI_STATUS_IGNORE);
    }

    void Session::ParsePlan(util::json j)
    {
        std::string host_id = j["host_id"];
        std::string device_id = j["device_id"];
        std::string num_peers = j["num_peers"];

        global_rank_ = static_cast<uint>(std::stoi(host_id));
        local_rank_ = static_cast<uint>(std::stoi(device_id));
        num_ranks_ = static_cast<uint>(std::stoi(num_peers));

        //TODO: configurable thru plan or env var, defaults to 16MB
        cuda_recv_buf_sze_ = 16 * 1024 * 1024;

        for (auto element : j["tasks"])
        {
            // LOG(INFO)<< element << '\n';
            std::string target_host_id = element["target_host_id"];
            std::string target_device_id = element["target_device_id"];
            std::string tensor_name = element["tensor_name"];
            std::string route_type = element["route_type"];

            if (route_type == "PCIE")
            {
                auto id = static_cast<uint>(std::stoi(target_device_id));
                if (std::find(pcie_targets_.begin(), pcie_targets_.end(), id) ==
                    pcie_targets_.end())
                    pcie_targets_.push_back(id);
            }
            else
            {
                auto id = static_cast<uint>(std::stoi(target_host_id));
                if (std::find(rdma_targets_.begin(), rdma_targets_.end(), id) ==
                    rdma_targets_.end())
                    rdma_targets_.push_back(id);
                rdma_targets_.push_back(static_cast<uint>(std::stoi(target_host_id)));
            }
            table_[tensor_name].push_back(element);
        }

#ifndef NDEBUG
        for (auto itr = pcie_targets_.begin(); itr != pcie_targets_.end(); itr++)
            LOG(INFO) << *itr << " ";
        for (auto itr = rdma_targets_.begin(); itr != rdma_targets_.end(); itr++)
            LOG(INFO) << *itr << " ";
#endif
    }

    template <class DataType>
    bool Session::is_expected_type(DataType* ptr, std::string op)
    {
        if (op == "DT_FLOAT")
            return typeid(*ptr) == typeid(float);

        else if (op == "DT_INT")
            return typeid(*ptr) == typeid(int);
        else
            return false;
    }

}; // namespace superscaler
