// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow
{
    class TensorProto;

    class GPUUtil
    {
    public:
        // "tensor" is GPU-local.  "dev" is the hosting GPU.
        // "device_context" should be the context of the GPU "_Send" op
        // which provides the Tensor.
        // Sets all necessary fields of "proto" by transferring value
        // bytes from GPU to CPU RAM. "is_dead" indicates that the
        // tensor is dead with an uninit value.
        static void SetProtoFromGPU(const Tensor& tensor,
                                    Device* dev,
                                    const DeviceContext* device_context,
                                    TensorProto* proto,
                                    bool is_dead,
                                    StatusCallback done);

        // Copies the data in 'gpu_tensor' into 'cpu_tensor'.
        // 'gpu_tensor''s backing memory must be on 'gpu_device' and
        // 'cpu_tensor' must be allocated to be of the same size as
        // 'gpu_tensor'. Synchronous: may block.
        static void CopyGPUTensorToCPU(Device* gpu_device,
                                       const DeviceContext* device_context,
                                       const Tensor* gpu_tensor,
                                       Tensor* cpu_tensor,
                                       StatusCallback done);

        // Blocks until all operations queued on the stream associated with
        // "gpu_device" at the time of the call have completed.  Returns any
        // error pending on the stream at completion.
        static Status Sync(Device* gpu_device);

        // Blocks until all operations queued on all streams associated with the
        // corresponding GPU device at the time of call have completed.
        // Returns any error pending on the stream at completion.
        static Status SyncAll(Device* gpu_device);

        // For debugging purpose, given a "device" and a "tensor" allocated
        // on the device, return a string printing each byte in the tensor
        // (up to a limit).  "device" can be either a CPU or a GPU device.
        static string MemoryDebugString(const Device* device, Tensor* tensor);

        // Map a Tensor as a DeviceMemory object wrapping the given typed
        // buffer.
        //
        // NOTE: will be removed soon, see StreamExecutorUtil::AsDeviceMemory
        // instead.
        template <typename T>
        static se::DeviceMemory<T> AsDeviceMemory(const Tensor& t)
        {
            T* ptr = reinterpret_cast<T*>(const_cast<void*>(DMAHelper::base(&t)));
            return se::DeviceMemory<T>(se::DeviceMemoryBase(ptr, t.TotalBytes()));
        }

        // Computes a checksum over the contents of "tensor", which is allocated
        // on "gpu_device".
        static uint64
            Checksum(Device* gpu_device, const DeviceContext* device_context, const Tensor& tensor);

        // Computes a checksum over the contents of "tensor", which is allocated
        // in local CPU RAM.
        static uint64 Checksum(const Tensor& tensor);

        static void CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                                       const DeviceContext* device_context,
                                       Device* gpu_device,
                                       Tensor* gpu_tensor,
                                       StatusCallback done,
                                       bool sync_dst_compute);

        static void DeviceToDeviceCopy(DeviceContext* send_dev_context,
                                       DeviceContext* recv_dev_context,
                                       Device* src,
                                       Device* dst,
                                       AllocatorAttributes src_alloc_attr,
                                       AllocatorAttributes dst_alloc_attr,
                                       const Tensor* input,
                                       Tensor* output,
                                       int dev_to_dev_stream_index,
                                       StatusCallback done);

        // Deep-copying of GPU tensor on the same device.
        // 'src_gpu_tensor''s and 'dst_gpu_tensor''s backing memory must be on
        // 'gpu_device' and 'dst_cpu_tensor' must be allocated to be of the same
        // size as 'src_gpu_tensor'.
        static void CopyGPUTensorToSameGPU(Device* gpu_device,
                                           const DeviceContext* device_context,
                                           const Tensor* src_gpu_tensor,
                                           Tensor* dst_gpu_tensor,
                                           StatusCallback done);
    };

} // namespace tensorflow
