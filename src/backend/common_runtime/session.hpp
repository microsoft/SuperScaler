#pragma once

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "executor_pub.hpp"
#include "util.hpp"

#define SC_DISABLE_COPY(Class)                                                                     \
    Class(const Class&);                                                                           \
    Class& operator=(const Class&);

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

        void Send(const char* tensor_name, unsigned char* input, size_t size);
        void Recv(const char* tensor_name, unsigned char** output, size_t* size);

        inline int GetWorldSize() { return num_ranks_; }
        inline int GetLocalRank() { return local_rank_; }
        inline int GetGlobalRank() { return global_rank_; }
    private:
        //parse plan from json format
        void ParsePlan(util::json);
        //check if data type matches the op description
        template <class DataType>
        bool is_expected_type(DataType* ptr, std::string op);

        uint local_rank_;
        uint global_rank_;
        uint num_ranks_;

        std::vector<uint> pcie_targets_;
        std::vector<uint> rdma_targets_;
        std::unique_ptr<PollExecutor> exec_;
        std::shared_ptr<Channel> cuda_channel_;
        std::shared_ptr<Channel> rdma_channel_;
        std::unordered_map<std::string, std::vector<util::json>> table_;

        //each sess will have a staging buffer for receiving data
        void* cuda_recv_buf_;
        size_t cuda_recv_buf_sze_;
        std::mutex sc_mu_;
        SC_DISABLE_COPY(Session);
    };

}; // namespace superscaler
