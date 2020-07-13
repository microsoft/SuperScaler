/**
 * This code is used for demostrating the use of executor
 * not compilable now
 * 
 * rewrite https://msrasrg.visualstudio.com/SuperScaler/_git/SuperScaler?path=%2Fsuperscaler_dll%2Fasync_comm%2Ftest%2Ftest_allreduce.cpp&version=GC1fbd30e9012ad8ed269b1d72f13346f822bc88e9
 * using current executor API
 */

#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>

#include "../task.hpp"
#include "../send_task.hpp"
#include "../recv_task.hpp"
#include "../rdma_tunnel.hpp"
#include "../poll_executor.hpp"
#include "../exec_info.hpp"

struct ReceiveContext {
	std::vector<int>	m_buffer;
	unsigned int		m_count;
};

template<class Channel, class Executor>
struct AllReduceContext {
	AllReduceContext(size_t tensor_size, const std::vector<unsigned int> &ranks)
	{
		// Init tensor
		for (size_t i = 0; i < ranks.size(); i++) {
			std::vector<int> tensor;
			std::generate_n(std::back_inserter(tensor), tensor_size, [=] { return rand() % tensor_size; });
			m_tensors.push_back(tensor);
			m_receive_context.push_back(ReceiveContext{std::vector<int>(tensor_size), 1});
		}
		/* init other members */
	}

	std::vector<std::vector<int> >	m_tensors;
	std::vector<ReceiveContext>		m_receive_context;
	Channel							m_channel;
	Executor						m_executor;
};

/**
 * @brief A sample task to compute weighted average on two 1-d tensors
 * 
 * @param tensor_0 average of \p count tensors
 * @param tensor_1 the tensor to be added to the average
 * @param count number of tensors that have been averaged
 */
class AverageTask : public Task {
public:
	AverageTask(Executor *exec, task_callback_t callback,
				std::vector<int> &tensor_0, const std::vector<int> &tensor_1,
				unsigned int &count)
		: Task(exec, callback), m_tensor_0(tensor_0), m_tensor_1(tensor_1),
		  m_count(count) {}

protected:
	TaskState execute(Executor *exec) override
	{
		auto itr0 = m_tensor_0.begin();
		auto itr1 = m_tensor_1.begin();
		unsigned int completed_count = ++m_count;
		std::generate_n(
			m_tensor_0.begin(), m_tensor_0.size(),
			[&itr0, &itr1, completed_count] {
				return (((double)completed_count - 1) / completed_count) * (*(itr0++))
				+ (1. / completed_count) * (*(itr1++));
			});
	}

private:
	std::vector<int> &m_tensor_0;
	const std::vector<int> &m_tensor_1;
	unsigned int &m_count;
};

/**
 * @brief A sample ring allreduce operation, consists of three tasks
 */
template<typename Channel>
static std::shared_ptr<Task> allreduce(
	AllReduceContext<Channel, PollExecutor> *ctx,
	unsigned int local_rank, unsigned peer_rank)
{	
	// create tasks

	// receive tensor to receive context from peer
	task_id_t recv_task = ctx->m_executor.create_task<RecvTask>(
		&(ctx->m_executor),
		[](TaskState) {},
		&(ctx->m_channel),
		local_rank,
		ctx->m_receive_context[local_rank].m_buffer.data(),
		ctx->m_receive_context[local_rank].m_buffer.size() * sizeof(ctx->m_receive_context[local_rank].m_buffer[0]));

	// update average on local tensor
	task_id_t average_task = ctx->m_executor.create_task<AverageTask>(
		&(ctx->m_executor),
		[](TaskState) {},
		ctx->m_tensors[local_rank],
		ctx->m_receive_context[local_rank].m_buffer,
		ctx->m_receive_context[local_rank].m_count);

	// send local tensor to peer
	task_id_t send_task = ctx->m_executor.create_task<SendTask>(
		&(ctx->m_executor),
		[](TaskState) {},
		&(ctx->m_channel),
		peer_rank,
		ctx->m_tensors[local_rank].data(),
		ctx->m_tensors[local_rank].size() * sizeof(ctx->m_tensors[local_rank][0]);

	/*
	 * establish dependence relationship
	 * 1. send tensor to peer node
	 * 2. receive tensor from peer node
	 * 3. compute average
	 */
	ctx->m_executor.add_dependence(recv_task, send_task);
	ctx->m_executor.add_dependence(average_task, recv_task);

	// add tasks to executor (the order of adding task doesn't matter)
	ctx->m_executor.add_task(recv_task);
	ctx->m_executor.add_task(send_task);
	ctx->m_executor.add_task(average_task);

	// return the last task of the operation
	return average_task;
}

int main()
{
	AllReduceContext<RDMAChannel, PollExecutor> ctx(100, {1, 0});

	// run allreduce on peer node 0
	auto allreduce_rank0 = allreduce(&ctx, 0, 1);
	// run allreduce on peer node 1
	auto allredcue_rank1 = allreduce(&ctx, 1, 0);

	// wait for the last task to finish
	auto ei0 = ctx.m_executor.wait(allreduce_rank0->get_task_id());
	auto ei1 = ctx.m_executor.wait(allredcue_rank1->get_task_id());
}
