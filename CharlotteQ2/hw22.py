#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:39:09 2017

@author: charlottejolloyd
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

data =  [2**6, 2**10, 2**16]

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
    
    __global__ void parallel_sum_gpu(float *in_data, float *out_data) {
        extern __shared__ float data[];
        unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
        data[tid] = in_data[tid];
        __syncthreads();
        for (unsigned int s=(blockDim.x*gridDim.x)/2; s>32; s>>=1) {
            if (tid < s) {
                data[tid] += data[tid + s];
            }
            __syncthreads();
        }
        if (tid < 32) {
            data[tid] += data[tid + 32];
            __syncthreads();
            data[tid] += data[tid + 16];
            __syncthreads();
            data[tid] += data[tid + 8];
            __syncthreads();
            data[tid] += data[tid + 4];
            __syncthreads();
            data[tid] += data[tid + 2];
            __syncthreads();
            data[tid] += data[tid + 1];
        }
        
        if (tid == 0) out_data[0] = data[0];
    }
    
    __global__ void getGlobalIdx_1D_1D(int *in_data) {
        int idx = threadIdx.x ;
        in_data[idx] = idx;
    }
    """)

func = mod.get_function("parallel_sum_gpu")
func2 = mod.get_function("getGlobalIdx_1D_1D")
#func = mod.get_function("doublify")


parallel_timings_gpu = []
start = 0
myarray = []

#for i in range(len(sizes)):
i=2
myarray = np.zeros((sizes[i],), dtype=np.int_) -70
in_data = myarray.astype(np.intc)
out_data = np.empty_like(in_data, dtype=np.intc)
#N = np.array(sizes[i], dtype=np.intc)


in_data_gpu = cuda.mem_alloc(in_data.nbytes)
#out_data_gpu = cuda.mem_alloc(out_data.nbytes)
#N_gpu = cuda.mem_alloc(N.nbytes)
cuda.memcpy_htod(in_data_gpu, in_data)
#cuda.memcpy_htod(out_data_gpu, out_data)
#cuda.memcpy_htod(N_gpu, N)

start = time.time()
func2(in_data_gpu, block=(1024,1,1), grid=(1024,1,1))
#func(in_data_gpu, out_data_gpu, block=(1,1,1), grid=(64,1,1))
cuda.memcpy_dtoh(out_data, in_data_gpu)
parallel_timings_gpu.append(time.time()-start)
print(out_data)


i=2
myarray = np.ones((sizes[i],), dtype=np.int_)
in_data = myarray.astype(np.float32)
out_data = np.zeros(1, dtype=np.float32)
#N = np.array(sizes[i], dtype=np.intc)


in_data_gpu = cuda.mem_alloc(in_data.nbytes)
out_data_gpu = cuda.mem_alloc(out_data.nbytes)
#N_gpu = cuda.mem_alloc(N.nbytes)
cuda.memcpy_htod(in_data_gpu, in_data)
cuda.memcpy_htod(out_data_gpu, out_data)
#cuda.memcpy_htod(N_gpu, N)

if sizes[i] <= 1024:
    xdim = sizes[i]
    ydim = 1
else:
    xdim = 1024
    if sizes[i]/1024 <= 1024:
        ydim = int(np.ceil(sizes[i]/1024))
    else:
        ydim = 1024
        zdim = 10

start = time.time()
func(in_data_gpu, out_data_gpu, block=(1024,1,1), grid=(1,1,1), shared=(in_data.nbytes)) #for some reason requesting 1 gpu core means I can't use more than 1024 here. Once i get to 1024 in any dimension, it throws an error for things beyond that. Probably something with the amount of gpus I can request or how many threads are available per gpu unit requested that i'm not familiar with.
cuda.memcpy_dtoh(out_data, out_data_gpu)
parallel_timings_gpu.append(time.time()-start)
print(out_data)
