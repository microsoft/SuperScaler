#ifndef HOROVOD_SUPER_OPERATIONS_H
#define HOROVOD_SUPER_OPERATIONS_H

#include <iostream>

#include "cuda_operations.h"
#include "super_scaler.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi/mpi_context.h"

namespace horovod {
namespace common {

class SuperAllreduce : public AllreduceOp {
public:
  SuperAllreduce(RdmaCommPrimitive* RdmaPrimitive, MPIContext* mpi_context,
                 HorovodGlobalState* global_state);

  virtual ~SuperAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
  RdmaCommPrimitive* RdmaPrimitive_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_SUPER_OPERATIONS_H
