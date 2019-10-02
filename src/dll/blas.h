#ifndef BLAS_H_
#define BLAS_H_

#include <cstdio>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <cmath>

#define BLOCK 512

inline dim3 cuda_gridsize(int n){
    int k = (n-1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {(uint)x,(uint) y, 1};
    return d;
}


void gradients_Average(float *gradients, int size, int nRanks);

void gradients_Reduce(float *gradients, float *buf, int size);


#endif // BLAS_H_