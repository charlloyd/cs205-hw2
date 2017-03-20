#!/usr/bin/env python
# -*- coding: utf-8 -*-
# adapted from: https://andreask.cs.illinois.edu/PyCuda/Examples/MatrixmulSimple
# adapted from: https://andreask.cs.illinois.edu/PyCuda/Examples/MatrixmulTiled

from __future__ import division
import numpy as np
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time
import csv

###########################################
### Define kernel functions in C
###########################################

simple_kernel_code_template = """
    __global__ void MatMulSimpleKernel(float *a, float *b, float *c)
    {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0;
    
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
    float Aelement = a[ty * %(MATRIX_SIZE)s + k];
    float Belement = b[k * %(MATRIX_SIZE)s + tx];
    Pvalue += Aelement * Belement;
    }
    
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
    }    
    """
block_kernel_code_template = """
    __global__ void MatMulBlockKernel(float *A, float *B, float *C)
    {
    const uint wA = %(MATRIX_SIZE)s;
    const uint wB = %(MATRIX_SIZE)s;
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    const uint aBegin = wA * %(BLOCK_SIZE)s * by;
    const uint aEnd = aBegin + wA - 1;
    const uint aStep = %(BLOCK_SIZE)s;
    const uint bBegin = %(BLOCK_SIZE)s * bx;
    const uint bStep = %(BLOCK_SIZE)s * wB;
    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
    __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
    __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
    
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];
    __syncthreads();
    
    for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
    Csub += As[ty][k] * Bs[k][tx];
    __syncthreads();
    }
    
    const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
    C[c + wB * ty + tx] = Csub;
    }        
    """

###########################################
### loop through code for various matrix sizes    
###########################################

MATRIX_SIZES = [2,2**2,2**3,2**4,2**5]
simple_t = []
block_t = []

for MATRIX_SIZE in MATRIX_SIZES:
    
    print("---------"+str(MATRIX_SIZE))

    # initialize matrices
    a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    #c_cpu = np.dot(a_cpu, b_cpu)
    
    # send matrices to GPU
    start = time.time()
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # compile & call function
    simple_kernel_code = simple_kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE
    }
    mod = compiler.SourceModule(simple_kernel_code)
    matmulsimple = mod.get_function("MatMulSimpleKernel")
    matmulsimple(a_gpu, b_gpu, c_gpu, block=(MATRIX_SIZE, MATRIX_SIZE, 1))

    print("---")
    print("SIMPLE")
    print("Matrix C (GPU):")
    print(c_gpu.get())
    simple_t.append(time.time()-start)
    
    print("Matrix A (GPU):")
    print(a_gpu.get())
    print("Matrix B (GPU):")
    print(b_gpu.get())
    #print("CPU-GPU difference:")
    #print(c_cpu - c_gpu.get())

    ###########################################
    
    TILE_SIZE = 2
    BLOCK_SIZE = TILE_SIZE

    # send matrices to GPU
    start = time.time()
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # compile & call function
    block_kernel_code = block_kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'BLOCK_SIZE': BLOCK_SIZE,
    }
    mod = compiler.SourceModule(block_kernel_code)
    matmulblock = mod.get_function("MatMulBlockKernel")
    matmulblock(a_gpu, b_gpu,c_gpu,grid = (MATRIX_SIZE // TILE_SIZE, MATRIX_SIZE // TILE_SIZE),block = (TILE_SIZE, TILE_SIZE, 1))

    print("---")
    print("BLOCK")
    print("Matrix C (GPU):")
    print(c_gpu.get())    
    block_t.append(time.time()-start)
    
    print("Matrix A (GPU):")
    print(a_gpu.get())
    print("Matrix B (GPU):")
    print(b_gpu.get())
    #print("CPU-GPU difference:")
    #print(c_cpu - c_gpu.get())
    #print("L2 norm:", la.norm(c_cpu - c_gpu.get()))
    
    #np.allclose(c_cpu, c_gpu.get())

print("---")
print(MATRIX_SIZES)
print(simple_t)
print(block_t)

with open('matmul.csv', 'a') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow([str(i) for i in MATRIX_SIZES])
    writer.writerow([str(i) for i in simple_t])
    writer.writerow([str(i) for i in block_t])
    f.close()
