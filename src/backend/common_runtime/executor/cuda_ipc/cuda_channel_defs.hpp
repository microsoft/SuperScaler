// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../channel/channel.hpp"

struct CudaTransferMeta {
    message_id_t id; /* Message id */
    cudaIpcMemHandle_t handler; /* Handler for receive buffer */
    size_t offset; /* Offset from the base address of handler */
    size_t length; /* except length */
};

struct CudaTransferAck {
    message_id_t id;
};