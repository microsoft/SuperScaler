// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "cuda_ipc/cuda_ipc_internal.hpp"
using compute_dev_id_t = int;
using compute_dev_stream_t = cudaStream_t;
#else
using compute_dev_id_t = int;
using compute_dev_stream_t = int;
#endif

struct ExecCtx {
    compute_dev_id_t compute_dev_id;
    compute_dev_stream_t compute_dev_stream;
};
