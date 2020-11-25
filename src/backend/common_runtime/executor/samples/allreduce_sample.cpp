// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <array>
#include <memory>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>

#include "../executor_pub.hpp"

#define GPU_0 0
#define GPU_1 1

constexpr message_id_t default_message_id = 0;
constexpr size_t test_size = 128;
std::vector<rank_t> devices = { GPU_0, GPU_1 };
std::array<int, test_size> tensor_0, tensor_1, expect;

void init_gpu_memory(void **cuda_recv, void **cuda_send, void *data, size_t size) {
	checkCudaErrors(cudaMalloc(cuda_recv, size));
	checkCudaErrors(cudaMalloc(cuda_send, size));
	checkCudaErrors(cudaMemcpy(*cuda_send, data, size, cudaMemcpyDefault));
}

void cleanup_gpu_memory(void *cuda_recv, void *cuda_send)
{
	checkCudaErrors(cudaFree(cuda_send));
	checkCudaErrors(cudaFree(cuda_recv));
}

void process_0()
{
	PollExecutor exec(GPU_0);
	auto chan = std::make_shared<CudaChannel>(GPU_0, devices);
	std::array<int, test_size> result;

	void *cuda_receive_buffer, *cuda_tensor_buffer;
	init_gpu_memory(&cuda_receive_buffer, &cuda_tensor_buffer,
					tensor_0.data(), test_size * sizeof(int));

	auto send_task = exec.create_task<SendTask>(&exec, nullptr,
												chan, GPU_1, default_message_id,
												cuda_tensor_buffer, test_size * sizeof(int));
	auto recv_task = exec.create_task<RecvTask>(&exec, nullptr,
												chan, GPU_1, default_message_id + 1,
												cuda_receive_buffer, test_size * sizeof(int));
	auto sum_task =
		exec.create_task<ReductionTask<int, SumKernelGPUImpl> >(
			&exec, nullptr, (int *)cuda_receive_buffer, (int *)cuda_tensor_buffer,
			SumKernelGPUImpl(), test_size);

	exec.add_dependence(sum_task, send_task);
	exec.add_dependence(sum_task, recv_task);

	exec.add_task(sum_task);
	exec.add_task(send_task);
	exec.add_task(recv_task);

	if (exec.wait(send_task).get_state() != ExecState::e_success) {
		std::cerr << "[Peer 0] Sender Error\n";
		goto clean_up;
	}
	if (exec.wait(recv_task).get_state() != ExecState::e_success) {
		std::cerr << "[Peer 0] Receiver Error\n";
		goto clean_up;
	}
	exec.wait(sum_task);

	checkCudaErrors(cudaMemcpy(result.data(), cuda_tensor_buffer,
							   test_size * sizeof(int), cudaMemcpyDefault));

	for (int i = 0; i < test_size; i++) {
		if (result[i] != expect[i]) {
			std::cerr << "[Peer 0] " << i << " th Data error\n";
			std::cerr << "Expected: " << expect[i]
					  << " Actual: " << result[i] << std::endl;
			goto clean_up;
		}
	}

	std::cerr << "[Peer 0] Success\n";
clean_up:
	cleanup_gpu_memory(cuda_receive_buffer, cuda_tensor_buffer);
	return;
}

void process_1()
{
	PollExecutor exec(GPU_1);
	auto chan = std::make_shared<CudaChannel>(GPU_1, devices);
	std::array<int, test_size> result;

	void *cuda_receive_buffer, *cuda_tensor_buffer;
	init_gpu_memory(&cuda_receive_buffer, &cuda_tensor_buffer,
					tensor_1.data(), test_size * sizeof(int));

	checkCudaErrors(cudaMemcpy(result.data(), cuda_tensor_buffer,
							   test_size * sizeof(int), cudaMemcpyDefault));

	auto send_task = exec.create_task<SendTask>(&exec, nullptr,
												chan, GPU_0, default_message_id + 1,
												cuda_tensor_buffer, test_size * sizeof(int));
	auto recv_task = exec.create_task<RecvTask>(&exec, nullptr,
												chan, GPU_0, default_message_id,
												cuda_receive_buffer, test_size * sizeof(int));
	auto sum_task =
		exec.create_task<ReductionTask<int, SumKernelGPUImpl> >(
			&exec, nullptr, (int *)cuda_receive_buffer, (int *)cuda_tensor_buffer,
			SumKernelGPUImpl(), test_size);

	exec.add_dependence(sum_task, send_task);
	exec.add_dependence(sum_task, recv_task);

	exec.add_task(sum_task);
	exec.add_task(send_task);
	exec.add_task(recv_task);

	if (exec.wait(send_task).get_state() != ExecState::e_success) {
		std::cerr << "[Peer 1] Sender Error\n";
		goto clean_up;
	}
	if (exec.wait(recv_task).get_state() != ExecState::e_success) {
		std::cerr << "[Peer 1] Receiver Error\n";
	}
	exec.wait(sum_task);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(result.data(), cuda_tensor_buffer,
							   test_size * sizeof(int), cudaMemcpyDefault));

	for (int i = 0; i < test_size; i++) {
		if (result[i] != expect[i]) {
			std::cerr << "[Peer 1] " << i << " th Data error\n";
			std::cerr << "Expected: " << expect[i]
					  << " Actual: " << result[i] << std::endl;
			goto clean_up;
		}
	}

	std::cerr << "[Peer 1] Success\n";
clean_up:
	cleanup_gpu_memory(cuda_receive_buffer, cuda_tensor_buffer);
	return;
}

int main()
{
	for (int i = 0; i < test_size; i++) {
		tensor_0[i] = rand() / 2;
		tensor_1[i] = rand() / 2;
		expect[i] = tensor_0[i] + tensor_1[i];
	}

	pid_t pid = fork();
	if (pid == 0) {
		process_1();
	} else {
		process_0();
		wait(NULL);
	}
	return 0;
}
