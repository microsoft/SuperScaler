// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "../cuda_ipc/cuda_channel.hpp"
#include "../send_task.hpp"
#include "../recv_task.hpp"
#include "../poll_executor.hpp"

constexpr message_id_t default_message_id = 0;
constexpr size_t test_size = 1024;
constexpr size_t num_test = 10;
std::vector<rank_t> devices = { 0, 1 };

void SenderProcess()
{
	auto chan = std::make_shared<CudaChannel>(0, devices);
	// Parper data
	std::vector<std::array<uint8_t, test_size> > input_datas(num_test);
	for (int n = 0; n < num_test; ++n)
		for (int i = 0; i < test_size; ++i)
			input_datas[n][i] = (i % 128) + n;

    uint8_t *cuda_input;
    checkCudaErrors(cudaMalloc(&cuda_input, test_size * num_test));

	for (int n = 0; n < num_test; ++n) {
		checkCudaErrors(cudaMemcpy(cuda_input + n * test_size, input_datas[n].data(), test_size,
								   cudaMemcpyDefault));

		auto send_task = std::make_shared<SendTask>(
			nullptr, nullptr, chan, 1, n, cuda_input + n * test_size, test_size);

		(*send_task)();
		if (send_task->get_state() != TaskState::e_success) {
			std::cerr << "[Sender Error] Message " << n << " failed\n";
			goto clean_up;
		}
	}
	std::cerr << "[Sender Info] Send success\n";
clean_up:
	checkCudaErrors(cudaFree(cuda_input));
}

void ReceiverProcess()
{
	auto chan = std::make_shared<CudaChannel>(1, devices);
	std::vector<std::array<uint8_t, test_size> > output_datas(num_test);

	uint8_t *cuda_output;
	checkCudaErrors(cudaMalloc(&cuda_output, test_size * num_test));
	checkCudaErrors(cudaMemset(cuda_output, 0, test_size * num_test));

	for (int n = 0; n < num_test; ++n) {
		auto recv_task = std::make_shared<RecvTask>(
			nullptr, nullptr, chan, 0, n, cuda_output + n * test_size, test_size);

		(*recv_task)();
		if (recv_task->get_state() != TaskState::e_success) {
			std::cerr << "[Receiver Error] Message " << n << " failed\n";
			goto clean_up;
		}
		checkCudaErrors(cudaMemcpy(output_datas[n].data(), cuda_output + n * test_size, test_size,
								cudaMemcpyDefault));
	}

	for (int n = 0; n < num_test; ++n)
		for (int i = 0; i < test_size; i++) {
			if (output_datas[n][i] != (i % 128) + n) {
				std::cerr << "[Receiver Error] " << n << " th test " << i << " th Data wrong\n";
				std::cerr << "Expect: " << (i % 128) + n
						  << " Actual: " << static_cast<int>(output_datas[n][i]) << std::endl;
				goto clean_up;
			}
		}
	std::cerr << "[Receiver Info] Receive success\n";
clean_up:
	checkCudaErrors(cudaFree(cuda_output));
}

int main()
{
	pid_t pid = fork();
	if (pid == 0) {
		SenderProcess();
	} else {
		ReceiverProcess();
		wait(NULL);
	}
	return 0;
}
