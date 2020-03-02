#ifndef SUPER_SCALAR_H_
#define SUPER_SCALAR_H_

#include "plan_parse/plan_parse.h"
#include "atomic_operations/comm_primitive.h"

// public function as interface
void initialization(int &myRank, int &nRanks, int &localRank, PlanTable &table);
void finalization();
int allReduce(float *gradients, size_t size, std::string tensorName, PlanTable table);

// private function
static uint64_t getHostHash(const char *string);
static void getHostName(char *hostname, int maxlen);
void allReduce_Ring(float *gradients, size_t size, std::string selfName, Plan plan);
void allReduce_Gradients_SumOrCover(float *gradients, int receiveAddress, float *receiveBuffer, int receiveLength, int type);
float *allReduce_Transmit_RDMA(RdmaCommPrimitive *rdmaCommPrimitive, float *gradients, int sendTarget, int sendAddress, int sendLength, int receiveTarget, int receiveAddress, int receiveLength);

#endif // SUPER_SCALAR_H_
