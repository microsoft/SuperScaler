// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
// #if HAVE_GPU
#include "tensorflow/stream_executor/stream.h"
// #endif
#include "gpu_util.hpp"
#include "session.hpp"

extern superscaler::Session sess;

namespace tensorflow
{
    REGISTER_OP("_SCSend")
        .Input("tensor: T")
        .Attr("T: type")
        .Attr("tensor_name: string")
        .Attr("send_device: string")
        .Attr("send_device_incarnation: string")
        .Attr("recv_device: string")
        .Attr("client_terminated: bool = false")
        .SetIsStateful()
        .SetShapeFn(shape_inference::UnknownShape);

    REGISTER_OP("_SCRecv")
        .Output("tensor: tensor_type")
        .Attr("tensor_type: type")
        .Attr("tensor_name: string")
        .Attr("send_device: string")
        .Attr("send_device_incarnation: string")
        .Attr("recv_device: string")
        .Attr("client_terminated: bool = false")
        .SetIsStateful()
        .SetShapeFn(shape_inference::UnknownShape);

    REGISTER_OP("_SCAllReduce")
        .Input("to_reduce: T")
        .Output("reduced: T")
        .Attr("T: {half, float, float64, int32, int64}")
        .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
        .Attr("num_devices: string")
        .Attr("tensor_name: string")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

    class SCAllReduceOp : public AsyncOpKernel
    {
    public:
        string GetCollectiveKey(OpKernelContext* c)
        {
            return strings::StrCat(collective_prefix_,
                                   ";",
                                   c->step_id(),
                                   ";",
                                   c->frame_iter().frame_id,
                                   ":",
                                   c->frame_iter().iter_id);
        }

        explicit SCAllReduceOp(OpKernelConstruction* context)
            : AsyncOpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("reduction", &reduction_));
            OP_REQUIRES_OK(context, context->GetAttr("num_devices", &num_devices_));
            OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &collective_prefix_));
        }

        void ComputeAsync(OpKernelContext* context, DoneCallback done) override
        {
            const Tensor* input_tensor = &context->input(0);
            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK_ASYNC(context,
                                 context->forward_input_or_allocate_output(
                                     {0}, 0, input_tensor->shape(), &output_tensor),
                                 done);

            auto device_context = context->op_device_context();
            auto gpu_id = context->device()->tensorflow_gpu_device_info()->gpu_id;

            VLOG(1) << "sc allreduce on device " << gpu_id;

            if (device_context != nullptr)
            {
                auto* stream = device_context->stream();
                stream->BlockHostUntilDone();
            }

            // auto *compute_stream = context->op_device_context()->stream();
            // auto *gpu_info = context->device()->tensorflow_gpu_device_info();
            // Grab the input tensor
            auto input = input_tensor->flat<float>();
            auto* in_cudaptr = input.data();

            auto output = output_tensor->flat<float>();
            auto* out_cudaptr = output.data();
            auto size = output.size();

            // DCHECK_EQ(in_cudaptr, out_cudaptr);

            // allreduce
            VLOG(1) << reduction_ << " superscaler async allreduce " << collective_prefix_
                    << " @: " << in_cudaptr << " size: " << size << " to "
                    << " @: " << out_cudaptr;

            std::function<void(const Status&)> cb = [context, done](const Status& s) {
                context->SetStatus(s);
                done();
            };

            sess.AllReduce(collective_prefix_.c_str(), out_cudaptr, size, nullptr);
            auto st = Status::OK();
            cb(Status::OK());
        }

    private:
        string reduction_;
        string num_devices_;
        string collective_prefix_;
    };

    class SCSendOp : public OpKernel
    {
    public:
        explicit SCSendOp(OpKernelConstruction* ctx)
            : OpKernel(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
            OP_REQUIRES_OK(ctx,
                           ctx->GetAttr("send_device_incarnation", (&send_device_incarnation)));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &key_prefix_));
        }

        void Compute(OpKernelContext* c)
        {
            VLOG(1) << "sc send _SCSend: " << key_prefix_
                    << " synchronously on: " << c->device()->name();
            // Grab the input tensor
            Rendezvous::Args args;
            args.device_context = c->op_device_context();
            args.alloc_attrs = c->input_alloc_attr(0);
            Device* src_dev = dynamic_cast<Device*>(c->device());
            Status s;
            const Tensor& val = c->input(0);
            bool is_dead = c->is_input_dead();
            // auto input = input_tensor.flat<float>();
            TensorProto val_proto;

            // Send the Tensor description and data in a single transfer
            if (src_dev->tensorflow_gpu_device_info() && (!args.alloc_attrs.on_host()))
            {
                Notification n;
                GPUUtil::SetProtoFromGPU(val,
                                         src_dev,
                                         args.device_context,
                                         &val_proto,
                                         is_dead,
                                         [&n, &s](const Status& s_) {
                                             s = s_;
                                             n.Notify();
                                         });
                n.WaitForNotification();
            }
            else
            {
                val.AsProtoTensorContent(&val_proto);
            }

            std::function<void(const Status&)> cb = [c](const Status& s) { c->SetStatus(s); };

            std::string send;
            val_proto.SerializeToString(&send);
            VLOG(1) << "sc send got size: " << send.size();
            sess.Send(key_prefix_.c_str(),
                      (unsigned char*)const_cast<char*>(send.c_str()),
                      send.size(),
                      nullptr);
            auto st = Status::OK();
            cb(st);
        }

    private:
        string send_device;
        string recv_device;
        string send_device_incarnation;
        string key_prefix_;

        string GetCollectiveKey(OpKernelContext* c)
        {
            return strings::StrCat(key_prefix_,
                                   ";",
                                   c->step_id(),
                                   ";",
                                   c->frame_iter().frame_id,
                                   ":",
                                   c->frame_iter().iter_id);
        }
        TF_DISALLOW_COPY_AND_ASSIGN(SCSendOp);
    };

    class SCRecvOp : public AsyncOpKernel
    {
    public:
        explicit SCRecvOp(OpKernelConstruction* ctx)
            : AsyncOpKernel(ctx)
        {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
            OP_REQUIRES_OK(ctx,
                           ctx->GetAttr("send_device_incarnation", (&send_device_incarnation)));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &key_prefix_));
        }
        void ComputeAsync(OpKernelContext* c, DoneCallback done)
        {
            VLOG(1) << "sc recv _SCRecv: " << key_prefix_
                    << " asynchronously on: " << c->device()->name();

            auto& worker_threads = *(c->device()->tensorflow_cpu_worker_threads());
            thread::ThreadPool* thread_pool = worker_threads.workers;

            VLOG(1) << "thread_pool size: " << thread_pool->NumThreads();
            VLOG(1) << "current thread id: " << thread_pool->CurrentThreadId();

            Rendezvous::Args args;
            args.device_context = c->op_device_context();
            args.alloc_attrs = c->output_alloc_attr(0);

            // Tensor output;
            // OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, shape, &output), done);
            char* buff;
            size_t size;
            sess.Recv(key_prefix_.c_str(), (unsigned char**)&buff, &size, nullptr);
            auto st = Status::OK();

            string recv(buff);
            delete buff;

            VLOG(1) << "sc recv got string size:  " << size;
            VLOG(1) << "restoring tensor from bytestream";

            TensorProto output_tensor_proto;
            // std::string recv;
            output_tensor_proto.ParseFromString(recv);

            VLOG(1) << "restoring done : " << key_prefix_;

            Tensor output;

            VLOG(1) << "making tensor : " << key_prefix_;

            // OP_REQUIRES_OK(c, c->device()->MakeTensorFromProto(proto,
            // c->output_alloc_attr(0), &output));
            OP_REQUIRES_OK_ASYNC(c,
                                 c->device()->MakeTensorFromProto(
                                     output_tensor_proto, c->output_alloc_attr(0), &output),
                                 done);

            VLOG(1) << "making tensor done : " << key_prefix_;

            std::function<void(const Status& s, const Tensor& val, bool is_dead)> cb =
                [c, done](const Status& s, const Tensor& val, bool is_dead) {
                    c->SetStatus(s);
                    if (s.ok())
                    {
                        if (!is_dead)
                        {
                            c->set_output(0, val);
                            VLOG(1) << "sc recv done : "
                                    << " asynchronously on: " << c->device()->name();
                        }
                    }
                    done();
                };
            cb(st, output, false);
        }

    private:
        string key_prefix_;
        string send_device;
        string recv_device;
        string send_device_incarnation;
        string GetCollectiveKey(OpKernelContext* c)
        {
            return strings::StrCat(key_prefix_,
                                   ";",
                                   c->step_id(),
                                   ";",
                                   c->frame_iter().frame_id,
                                   ":",
                                   c->frame_iter().iter_id);
        }
        TF_DISALLOW_COPY_AND_ASSIGN(SCRecvOp);
    };

    REGISTER_KERNEL_BUILDER(Name("_SCSend").Device(DEVICE_CPU), SCSendOp);
    REGISTER_KERNEL_BUILDER(Name("_SCSend").Device(DEVICE_GPU), SCSendOp);
    REGISTER_KERNEL_BUILDER(Name("_SCRecv").Device(DEVICE_CPU), SCRecvOp);
    REGISTER_KERNEL_BUILDER(Name("_SCRecv").Device(DEVICE_GPU), SCRecvOp);
    REGISTER_KERNEL_BUILDER(Name("_SCAllReduce").Device(DEVICE_CPU), SCAllReduceOp);
    REGISTER_KERNEL_BUILDER(Name("_SCAllReduce").Device(DEVICE_GPU), SCAllReduceOp);

} // namespace tensorflow
