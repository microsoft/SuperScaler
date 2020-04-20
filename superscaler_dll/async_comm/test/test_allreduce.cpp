#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <exception>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <gtest/gtest.h>

#include <task.hpp>
#include <worker.hpp>
#include <poll_executor.hpp>
#include <rdma_task.hpp>
#include <rdma_tunnel.hpp>

struct ReceiveContext {
    std::vector<int> m_buffer;
    unsigned int     m_count;
};

template<class Channel>
struct AllReduceContext {
public:

    AllReduceContext(
        size_t tensor_size,
        const std::vector<unsigned int> & ranks,
        size_t worker_size = std::numeric_limits<size_t>::max()) :
        m_channel(tensor_size * sizeof(m_tensors[0][0]), ranks),
        m_exec(worker_size),
        m_sent_completed_count(0) {
        // Init tensor
        for (size_t i = 0; i < ranks.size(); ++i) {
            std::vector<int> tensor;
            std::generate_n(std::back_inserter(tensor), tensor_size, [=] { return rand() % tensor_size; });
            m_tensors.push_back(tensor);
            m_receive_context.push_back(ReceiveContext{std::vector<int>(tensor_size), 1});
        }
        // Calculate result
        for (size_t i = 0; i < tensor_size; ++i) {
            int target = 0;
            for (size_t j = 0; j < ranks.size(); ++j) {
                target += m_tensors[j][i];
            }
            target /= ranks.size();
            m_result.push_back(target);
        }
    }

    void check_result() const {
        for (auto & tensor : m_tensors) {
            ASSERT_EQ(tensor, m_result);
        }
    }

    std::vector<std::vector<int> >  m_tensors;
    std::vector<ReceiveContext>     m_receive_context;
    std::vector<int>                m_result;
    Channel                         m_channel;
    PollExecutor                    m_exec;
    std::mutex                      m_mutex;
    std::condition_variable         m_condition;
    std::atomic<unsigned int>       m_sent_completed_count;
};

template<typename Channel, typename SendTask, typename ReceiveTask>
static std::shared_ptr<Task> allreduce(
    AllReduceContext<Channel> * ctx,
    unsigned int local_rank,
    unsigned int peer_rank) {

    if (ctx == nullptr) {
        throw std::invalid_argument("Argument error");
    }

    auto recv_task = std::make_shared<ReceiveTask>(
        &(ctx->m_exec),
        [ctx, local_rank] {
            // Do average task
            auto itr0 = ctx->m_tensors[local_rank].begin();
            auto itr1 = ctx->m_receive_context[local_rank].m_buffer.begin();
            unsigned int receive_completed_count = ++(ctx->m_receive_context[local_rank].m_count);
            std::generate_n(
                ctx->m_tensors[local_rank].begin(),
                ctx->m_tensors[local_rank].size(), 
                [&itr0, &itr1, receive_completed_count] {
                    return 
                        (((double)receive_completed_count - 1) / receive_completed_count)*(*(itr0++)) 
                        + (1.0/receive_completed_count)*(*(itr1++));
                });
        },
        &(ctx->m_channel),
        local_rank,
        ctx->m_receive_context[local_rank].m_buffer.data(),
        ctx->m_receive_context[local_rank].m_buffer.size() * sizeof(ctx->m_receive_context[local_rank].m_buffer[0])
    );

    auto send_task = std::make_shared<SendTask>(
        &(ctx->m_exec),
        [ctx, recv_task] {

            // Wait all send finish
            ctx->m_sent_completed_count++;
            if (ctx->m_sent_completed_count == ctx->m_tensors.size()) {
                ctx->m_condition.notify_all();
            } else {
                std::unique_lock<std::mutex> lock(ctx->m_mutex);
                ctx->m_condition.wait(lock, [&ctx] {
                    return ctx->m_sent_completed_count == ctx->m_tensors.size();
                });
            }

            ctx->m_exec.add_task(recv_task, true);
        },
        &(ctx->m_channel),
        peer_rank,
        ctx->m_tensors[local_rank].data(),
        ctx->m_tensors[local_rank].size() * sizeof(ctx->m_tensors[local_rank][0]));
    ctx->m_exec.add_task(send_task, true);

    return recv_task;
}

TEST(AllReduce, PureRDMASingle) {
    AllReduceContext<RDMAChannel> ctx(100, {0});

    auto allreduce_rank0 = allreduce<RDMAChannel, RDMASendTask, RDMAReceiveTask>(&ctx, 0, 0);
    allreduce_rank0->wait();

    ctx.check_result();
}

TEST(AllReduce, PureRDMADouble) {
    AllReduceContext<RDMAChannel> ctx(100, {0, 1});

    auto allreduce_rank0 = allreduce<RDMAChannel, RDMASendTask, RDMAReceiveTask>(&ctx, 0, 1);
    auto allreduce_rank1 = allreduce<RDMAChannel, RDMASendTask, RDMAReceiveTask>(&ctx, 1, 0);    
    allreduce_rank0->wait();
    allreduce_rank1->wait();

    ctx.check_result();
}

