#include "super_operations.h"

namespace horovod {
namespace common {

SuperAllreduce::SuperAllreduce(RdmaCommPrimitive* RdmaPrimitive,
                               MPIContext* mpi_context,
                               HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mpi_context_(mpi_context),
      RdmaPrimitive_(RdmaPrimitive) {}

Status SuperAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                               const Response& response) {

  auto& first_entry = entries[0];

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // CfgTable table = RdmaPrimitive_->get_rdma_cfg();

  int rank = global_state_->controller->GetRank();
  int nRanks = global_state_->controller->GetSize();
  int localRank = global_state_->controller->GetLocalRank();

  const void* fused_input_data;
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
  } else {
    buffer_data = (void*)first_entry.output->data();
    buffer_len = (size_t)first_entry.output->size();
    fused_input_data = buffer_data;
  }

  const void* sendbuf = MPI_IN_PLACE;

  auto& timeline = global_state_->timeline;

  // RdmaPrimitive_->RDMA_Register_CPU_MemRegion((float*)buffer_data,num_elements);

  // if (table.has_plan(first_entry.tensor_name)) {
  //   auto plan = table.get_plan(first_entry.tensor_name);
  //   // LOG(INFO, rank) << "\nSuper_Allreduce cfg_table find " <<
  //   // first_entry.tensor_name << "with size = " << num_elements;
  //   for (auto op_ : plan.get_operations()) {
  //     RdmaPrimitive_->run_write_host((float*)buffer_data, num_elements, rank,
  //                                    nRanks, localRank, op_);
  //   }
  // } else {
  //   int op =
  //       MPI_Allreduce(MPI_IN_PLACE, buffer_data, (int)num_elements,
  //                     mpi_context_->GetMPIDataType(first_entry.tensor),
  //                     mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
  //                     mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  // }


////////////////////////////////////////////////////////////////////////////////////////////
    size_t size = 512;
    float *gradients = new float[size];
    for (int i = 0; i < size; i++){
        gradients[i] = i;
    }
    
    RdmaPrimitive_->RDMA_Register_CPU_MemRegion(gradients, size);

    auto plan = RdmaPrimitive_->get_rdma_cfg().cfg_table["allreduce.classifier.6.bias"];

    for(auto op_:plan.operation)
    {										  
        RdmaPrimitive_->execute(gradients, size, rank, nRanks, localRank, op_);
    }
    // for (int k = 0; k < size; k++){
    //     gradients[k] /= nRanks;
    // }
    free(gradients);
//////////////////////////////////////////////////////////////////////////////







  // auto plan = RdmaPrimitive_->get_rdma_cfg().cfg_table["allreduce.classifier.6.bias"];
  // for(auto op_:plan.operation){						


  // std::cout << "===============================================================" << std::endl;
  // std::cout << "num_elements: " << num_elements << std::endl;
  // std::cout << "rank: " << rank << " " << nRanks<< " " << localRank << " " << std::endl;
  // std::cout << "===============================================================" << std::endl;


  //          RdmaPrimitive_->run_write_host((float*)buffer_data, num_elements, rank,
  //                                    nRanks, localRank, op_);
  
  
  
  // std::cout << "===============================================================" << std::endl;
  // std::cout << "after run_write_host" << std::endl;
  // std::cout << "===============================================================" << std::endl;
        
        
        
  //       }
  //       // for (int k = 0; k < size; k++){
  //       //     gradients[k] /= nRanks;
  //       // }




  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);
  }

  return Status::OK();
}

bool SuperAllreduce::Enabled(const ParameterManager& param_manager,
                             const std::vector<TensorTableEntry>& entries,
                             const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
