// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "superscaler_pywrap.hpp"
#include "session.hpp"

superscaler::Session sess;

#ifdef __cplusplus
extern "C" {
#endif

//sess is shared global per tf process
void sc_init(const char* plan_path)
{
    sess.Create(plan_path);
}

void sc_finalize()
{
    sess.Close();
}

void sc_get_world_size(int* size)
{
    *size = sess.GetWorldSize();
}

void sc_get_device_id(int* device_id)
{
    *device_id = sess.GetDeviceId();
}

void sc_get_host_id(int* host_id)
{
    *host_id = sess.GetHostId();
}

// in-place allreduce
void sc_allreduce(const char* tensor_name, float* data, size_t size, void* stream)
{
    sess.AllReduce(tensor_name, data, size, stream);
}

void sc_send(const char* tensor_name, unsigned char* input, size_t size, void* stream)
{
    sess.Send(tensor_name, input, size);
}

void sc_recv(const char* tensor_name, unsigned char** output, size_t* size, void* stream)
{
    sess.Recv(tensor_name, output, size);
}

#ifdef __cplusplus
}
#endif