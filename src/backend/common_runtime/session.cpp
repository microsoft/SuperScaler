// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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
        VLOG(1) << "superscaler session is created from " << std::string(plan_fpath);
        auto plan = util::JsonParser::load_from(plan_fpath);
        Create(plan);
    }

    void Session::Create(util::json j)
    {
        ParsePlan(j);
        checkCudaErrors(cudaSetDevice(device_id_));
        checkCudaErrors(cudaMalloc(&recv_buf_, recv_buf_sze_));
        exec_.reset(new PollExecutor(device_id_));
        //TODO: add rdma channels support
        cuda_channel_ = std::make_shared<CudaChannel>(static_cast<uint>(device_id_), pcie_targets_);
    }

    void Session::Close()
    {
        //those executor-related instances(especially one with cuda-related calls) needs to be destruct explicitly before cuda_recv_buf_ is freed
        //to avoid cuda driver shutting down error
        exec_.reset();
        cuda_channel_.reset();
        rdma_channel_.reset();
        table_.clear();

        checkCudaErrors(cudaSetDevice(device_id_));
        if (recv_buf_)
            checkCudaErrors(cudaFree(recv_buf_));
        VLOG(1) << "superscaler session is destoryed";
    }

    template <class DataType>
    void Session::AllReduce(const char* tensor_name, DataType* ioput, size_t size, void* stream)
    {
#ifndef NDEBUG
        VLOG(1) << "[" << host_id_ << ":" << device_id_ << "]: allreduce " << tensor_name << " @ "
                << ioput << " with size: " << size;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
        auto tasks = table_[tensor_name];
        // #ifndef NDEBUG
        VLOG(1) << tensor_name << " got " << tasks.size() << " plans to run";
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
                    VLOG(1) << "[" << host_id_ << "/" << device_id_ << "]: send " << msg_id << " @ "
                            << ioput + offset << " with size: " << sze << " to " << target_id;
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

                    std::string reduction_op = element["reduction"];

                    if (reduction_op == "sum")
                    {
                        VLOG(1) << "[" << host_id_ << "/" << device_id_ << "]: recv " << msg_id
                                << " @ " << recv_buf_ << " with size: " << sze << " from "
                                << target_id;
                        ;
                        recv_task_id = exec_->create_task<RecvTask>(exec_.get(),
                                                                    nullptr,
                                                                    cuda_channel_,
                                                                    target_id,
                                                                    msg_id,
                                                                    recv_buf_,
                                                                    sze * sizeof(DataType));

                        VLOG(1) << "[" << host_id_ << "/" << device_id_ << "]: reduce " << msg_id
                                << " @ " << ioput + offset << " with size: " << sze;
                        reduce_task_id =
                            exec_->create_task<ReductionTask<DataType, SumKernelGPUImpl>>(
                                exec_.get(),
                                [&](TaskState) {
                                    cudaStreamSynchronize(exec_->get_context()->compute_dev_stream);
                                },
                                (const DataType*)recv_buf_,
                                ioput + offset,
                                SumKernelGPUImpl(),
                                sze);

                        exec_->add_dependence(reduce_task_id, recv_task_id);
                        exec_->add_task(send_task_id);
                        exec_->add_task(recv_task_id);
                        exec_->add_task(reduce_task_id);
                        if (exec_->wait(send_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Send error\n", device_id_);
                        }
                        if (exec_->wait(recv_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Recv error\n", device_id_);
                        }
                        if (exec_->wait(reduce_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Reduce error\n", device_id_);
                        }
                    }
                    else
                    {
                        VLOG(1) << "[" << host_id_ << "/" << device_id_ << "]: recv " << msg_id
                                << " @ " << ioput + offset << " with size: " << sze << " from "
                                << target_id;
                        ;
                        recv_task_id = exec_->create_task<RecvTask>(exec_.get(),
                                                                    nullptr,
                                                                    cuda_channel_,
                                                                    target_id,
                                                                    msg_id,
                                                                    ioput + offset,
                                                                    sze * sizeof(DataType));

                        exec_->add_task(send_task_id);
                        exec_->add_task(recv_task_id);
                        if (exec_->wait(send_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Send error\n", device_id_);
                        }
                        if (exec_->wait(recv_task_id).get_state() != ExecState::e_success)
                        {
                            fprintf(stderr, "[Peer %d] Recv error\n", device_id_);
                        }
                    }
                }
            }
            else
            {
                //TODO: RDMA
                throw std::runtime_error("unknown route type, not supported yet!");
            }
        }
        //avg
        ScaleTask<DataType, DivKernelGPUImpl> scale_task(
            exec_.get(),
            [&](TaskState) { cudaStreamSynchronize(exec_->get_context()->compute_dev_stream); },
            ioput,
            num_participants_,
            DivKernelGPUImpl(),
            size);
        scale_task();
        if (scale_task.get_state() != TaskState::e_success)
            fprintf(stderr, "[Peer %d] Div error\n", device_id_);

#ifndef NDEBUG
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        VLOG(1) << " with time elapsed: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                << " [µs]";
#endif
    }

    void Session::Send(const char* tensor_name, unsigned char* input, size_t size, void* stream)
    {
        //TODO: add send implementation
        throw std::runtime_error("not supported yet!");
    }

    void Session::Recv(const char* tensor_name, unsigned char** output, size_t* size, void* stream)
    {
        //TODO: add recv implementation
        throw std::runtime_error("not supported yet!");
    }

    void Session::ParsePlan(util::json j)
    {
        std::string host_id = j["host_id"];
        std::string device_id = j["device_id"];
        std::string num_peers = j["num_peers"];

        host_id_ = static_cast<uint>(std::stoi(host_id));
        device_id_ = static_cast<uint>(std::stoi(device_id));
        num_participants_ = static_cast<uint>(std::stoi(num_peers));

        //configurable thru plan or env var, defaults to 256MB
        recv_buf_sze_ = j.contains("recv_buffer_size") ? j["recv_buffer_size"].get<size_t>()
                                                       : 256 * 1024 * 1024;
        VLOG(1) << "recv_buf_sze_ : " << recv_buf_sze_;

        for (auto element : j["tasks"])
        {
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
                //TODO: rdma targets
                throw std::runtime_error("unknown route type, not supported yet!");
            }
            table_[tensor_name].push_back(element);
        }
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

    template void Session::AllReduce(const char* tensor_name,
                                     double* ioput,
                                     size_t size,
                                     void* stream = nullptr);
    template void Session::AllReduce(const char* tensor_name,
                                     float* ioput,
                                     size_t size,
                                     void* stream = nullptr);
    template void Session::AllReduce(const char* tensor_name,
                                     int* ioput,
                                     size_t size,
                                     void* stream = nullptr);

}; // namespace superscaler
