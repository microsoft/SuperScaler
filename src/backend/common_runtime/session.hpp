// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "executor_pub.hpp"
#include "util.hpp"
namespace superscaler
{
    class Session
    {
    public:
        Session() {}
        void Create(const char* plan_fpath);
        void Create(util::json j);
        void Close();
        template <class DataType>
        void AllReduce(const char* tensor_name,
                       DataType* ioput,
                       size_t size,
                       void* stream = nullptr);

        void Send(const char* tensor_name,
                  unsigned char* input,
                  size_t size,
                  void* stream = nullptr);
        void Recv(const char* tensor_name,
                  unsigned char** output,
                  size_t* size,
                  void* stream = nullptr);

        inline int GetWorldSize() { return num_participants_; }
        inline int GetDeviceId() { return device_id_; }
        inline int GetHostId() { return host_id_; }

    private:
        //parse plan from json format
        void ParsePlan(util::json);
        //check if data type matches the op description
        template <class DataType>
        bool is_expected_type(DataType* ptr, std::string op);

        uint device_id_;
        uint host_id_;
        uint num_participants_;

        std::vector<uint> pcie_targets_;
        std::vector<uint> rdma_targets_;
        std::unique_ptr<PollExecutor> exec_;
        std::shared_ptr<Channel> cuda_channel_;
        std::shared_ptr<Channel> rdma_channel_;
        std::unordered_map<std::string, std::vector<util::json>> table_;

        //each sess will have a staging buffer for receiving data
        void* recv_buf_;
        size_t recv_buf_sze_;
        std::mutex sc_mu_;
        SC_DISALLOW_COPY_AND_ASSIGN(Session);
    };

}; // namespace superscaler
