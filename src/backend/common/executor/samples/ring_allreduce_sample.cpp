#include <memory>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>

#include "../executor_pub.hpp"

template <class DataType, class ReduceKernel>
int ring_allreduce_worker(compute_dev_id_t self_compute_dev,
                          compute_dev_id_t prev_peer_compute_dev,
                          compute_dev_id_t next_peer_compute_dev,
                          uint64_t self_compute_dev_idx,
                          uint64_t num_compute_devs,
                          DataType* cpu_data_buf,
                          uint64_t num_elements,
                          uint64_t num_loop)
{
    PollExecutor exec(self_compute_dev);
    std::vector<rank_t> peer_devs =
        {static_cast<rank_t>(prev_peer_compute_dev),
         static_cast<rank_t>(next_peer_compute_dev)};
    auto cuda_channel = std::make_shared<CudaChannel>(
        self_compute_dev, peer_devs);
    int error_ret = -1;
    int ret = 0;
    DataType *cuda_data_buf = nullptr;
    DataType *cuda_recv_buf = nullptr;
    uint64_t i = 0;
    uint64_t j = 0;
    task_id_t send_task_id = 0;
    task_id_t recv_task_id = 0;
    task_id_t reduce_task_id = 0;
    message_id_t send_msg_id = 0;
    message_id_t recv_msg_id = 0;
    uint64_t send_data_offset = 0;
    uint64_t recv_data_offset = 0;
    uint64_t reduce_data_offset = 0;
    uint64_t num_chunk_elements = 0;

    if (!(num_compute_devs > 1 && num_elements > 0 &&
        num_elements % num_compute_devs == 0)) {
        fprintf(
            stderr,
            "[Peer %lu] num_compute_devs should be > 1, "
            "num_elements should be > 0 and "
            "divisible by num_compute_devs.\n", self_compute_dev_idx);
        return error_ret;
    }
    num_chunk_elements = num_elements / num_compute_devs;

    // Allocate GPU buffer
    checkCudaErrors(cudaMalloc(
        &cuda_data_buf, num_elements * sizeof(DataType)));
    checkCudaErrors(cudaMalloc(
        &cuda_recv_buf, num_chunk_elements * sizeof(DataType)));

    // Copy data to GPU
    checkCudaErrors(cudaMemcpy(
        cuda_data_buf, cpu_data_buf, num_elements * sizeof(DataType),
        cudaMemcpyDefault));

    for (i = 0; i < num_loop; i++) {
        // Scatter-reduce
        for (j = 0; j < num_compute_devs - 1; j++) {
            send_data_offset =
                ((self_compute_dev_idx + num_compute_devs - j) %
                 num_compute_devs) * num_chunk_elements;
            send_task_id = exec.create_task<SendTask>(
                &exec, nullptr, cuda_channel, next_peer_compute_dev,
                send_msg_id, cuda_data_buf + send_data_offset,
                num_chunk_elements * sizeof(DataType));

            recv_task_id = exec.create_task<RecvTask>(
                &exec, nullptr, cuda_channel, prev_peer_compute_dev,
                recv_msg_id, cuda_recv_buf,
                num_chunk_elements * sizeof(DataType));

            reduce_data_offset =
                ((self_compute_dev_idx + num_compute_devs - j - 1) %
                 num_compute_devs) * num_chunk_elements;
            reduce_task_id =
                exec.create_task<ReductionTask<DataType, ReduceKernel>>(
                    &exec, nullptr,
                    cuda_recv_buf,
                    cuda_data_buf + reduce_data_offset,
                    ReduceKernel(), num_chunk_elements);

            exec.add_dependence(reduce_task_id, recv_task_id);

            exec.add_task(send_task_id);
            exec.add_task(recv_task_id);
            exec.add_task(reduce_task_id);

            if (exec.wait(send_task_id).get_state() != ExecState::e_success) {
                fprintf(stderr, "[Peer %lu] Send error\n",
                    self_compute_dev_idx);
                ret = error_ret;
                goto clean_up;
            }
            if (exec.wait(recv_task_id).get_state() != ExecState::e_success) {
                fprintf(stderr, "[Peer %lu] Receive error\n",
                    self_compute_dev_idx);
                ret = error_ret;
                goto clean_up;
            }
            if (exec.wait(reduce_task_id).get_state() != ExecState::e_success) {
                fprintf(stderr, "[Peer %lu] Reduce error\n",
                    self_compute_dev_idx);
                ret = error_ret;
                goto clean_up;
            }
        }

        // All-gather
        for (j = 0; j < num_compute_devs - 1; j++) {
            send_data_offset =
                ((self_compute_dev_idx + num_compute_devs - j + 1) %
                 num_compute_devs) * num_chunk_elements;
            send_task_id = exec.create_task<SendTask>(
                &exec, nullptr, cuda_channel, next_peer_compute_dev,
                send_msg_id, cuda_data_buf + send_data_offset,
                num_chunk_elements * sizeof(DataType));

            recv_data_offset =
                ((self_compute_dev_idx + num_compute_devs - j) %
                 num_compute_devs) * num_chunk_elements;
            // Use MemBlock here to make sure both base address and offset
            // are passed in.
            // TODO: refactor RecvTask interface to make it aware of non-base
            // CUDA addresses.
            recv_task_id = exec.create_task<RecvTask>(
                &exec, nullptr, cuda_channel, prev_peer_compute_dev,
                recv_msg_id,
                MemBlock(
                    cuda_data_buf, recv_data_offset * sizeof(DataType),
                    num_chunk_elements * sizeof(DataType)));

            exec.add_task(send_task_id);
            exec.add_task(recv_task_id);

            if (exec.wait(send_task_id).get_state() != ExecState::e_success) {
                fprintf(stderr, "[Peer %lu] Send error\n",
                    self_compute_dev_idx);
                ret = error_ret;
                goto clean_up;
            }
            if (exec.wait(recv_task_id).get_state() != ExecState::e_success) {
                fprintf(stderr, "[Peer %lu] Receive error\n",
                    self_compute_dev_idx);
                ret = error_ret;
                goto clean_up;
            }
        }
    }

    // Copy data from GPU
    checkCudaErrors(cudaMemcpy(
        cpu_data_buf, cuda_data_buf, num_elements * sizeof(DataType),
        cudaMemcpyDefault));

clean_up:
    // Clean GPU buffer
    checkCudaErrors(cudaFree(cuda_data_buf));
    checkCudaErrors(cudaFree(cuda_recv_buf));
    return ret;
}

int main(int argc, char** argv)
{
    constexpr uint64_t num_cuda_devs = 8;
    compute_dev_id_t cuda_dev_ids[num_cuda_devs] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint64_t num_elements = 128;
    uint64_t num_loop = 1;
    std::vector<float> cpu_data_buf;
    std::vector<float> expected_results;
    size_t i = 0;
    pid_t pid = 0;
    uint64_t self_compute_dev_idx = 0;
    int ret = 0;

    for (i = 1; i < num_cuda_devs; i++) {
        pid = fork();
        if (pid == 0) {
            self_compute_dev_idx = i;
            break;
        }
    }

    cpu_data_buf.resize(num_elements);
    expected_results.resize(num_elements);
    for (i = 0; i < num_elements; i++) {
        cpu_data_buf[i] = static_cast<double>(i);
        expected_results[i] = static_cast<double>(i * num_cuda_devs);
    }

    ret = ring_allreduce_worker<float, SumKernelGPUImpl>(
        cuda_dev_ids[self_compute_dev_idx],
        cuda_dev_ids[
            (self_compute_dev_idx + num_cuda_devs - 1) % num_cuda_devs],
        cuda_dev_ids[(self_compute_dev_idx + 1) % num_cuda_devs],
        self_compute_dev_idx,
        num_cuda_devs,
        cpu_data_buf.data(),
        num_elements,
        num_loop);

    if (ret) {
        fprintf(stderr, "[Peer %lu] ring_allreduce_worker error\n",
            self_compute_dev_idx);
        return ret;
    }

    // Should disable result checking if num_loop > 1
    for (i = 0; i < num_elements; i++) {
        if (std::abs(cpu_data_buf[i] - expected_results[i]) > 1e-6) {
            fprintf(stderr,
                "[Peer %lu] %lu-th data error, "
                "expected: %g, actual: %g\n",
                self_compute_dev_idx, i, expected_results[i], cpu_data_buf[i]);
        }
    }
    fprintf(stdout, "[Peer %lu] Success\n", self_compute_dev_idx);

    return 0;
}
