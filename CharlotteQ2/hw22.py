#!/usr/bin/env python
# -*- coding: utf-8 -*-
# copied from: https://andreask.cs.illinois.edu/PyCuda/Examples/MatrixmulSimple
# copied from: https://andreask.cs.illinois.edu/PyCuda/Examples/MatrixmulTiled

from __future__ import division
import numpy as np
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

###########################################
### Define kernel functions in ... C?
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
    
    // Block index
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    
    // Thread index
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by the block
    const uint aBegin = wA * %(BLOCK_SIZE)s * by;
    // Index of the last sub-matrix of A processed by the block
    const uint aEnd = aBegin + wA - 1;
    // Step size used to iterate through the sub-matrices of A
    const uint aStep = %(BLOCK_SIZE)s;
    
    // Index of the first sub-matrix of B processed by the block
    const uint bBegin = %(BLOCK_SIZE)s * bx;
    // Step size used to iterate through the sub-matrices of B
    const uint bStep = %(BLOCK_SIZE)s * wB;
    
    // The element of the block sub-matrix that is computed by the thread
    float Csub = 0;
    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
    a <= aEnd;
    a += aStep, b += bStep)
    {
    // Shared memory for the sub-matrix of A
    __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
    // Shared memory for the sub-matrix of B
    __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
    
    // Load the matrices from global memory to shared memory each thread loads one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    
    // Multiply the two matrices together; each thread computes one element of the block sub-matrix
    for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
    Csub += As[ty][k] * Bs[k][tx];
    
    // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
    }
    
    // Write the block sub-matrix to global memory; each thread writes one element
    const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
    C[c + wB * ty + tx] = Csub;
    }        
    """

###########################################
### loop through code for various matrix sizes    
###########################################

MATRIX_SIZES = [8,16,32,64,1024]

for MATRIX_SIZE in MATRIX_SIZES:
    
    print(MATRIX_SIZE)

    # initialize matrices
    a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    c_cpu = np.dot(a_cpu, b_cpu)

    # send matrices to GPU
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # compile & call function
    simple_kernel_code = simple_kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE
    }
    mod = compiler.SourceModule(simple_kernel_code)
    matmuls = mod.get_function("MatMulSimpleKernel")
    #matmuls(a_gpu, b_gpu, c_gpu, block=(MATRIX_SIZE, MATRIX_SIZE, 1))

    print("Matrix A (GPU):")
    #print(a_gpu.get())
    print("Matrix B (GPU):")
    #print(b_gpu.get())
    print("Matrix C (GPU):")
    #print(c_gpu.get())
    print("CPU-GPU difference:")
    #print(c_cpu - c_gpu.get())

    ###########################################
    
    TILE_SIZE = 2
    BLOCK_SIZE = TILE_SIZE

    # send matrices to GPU
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # compile & call function
    block_kernel_code = block_kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'BLOCK_SIZE': BLOCK_SIZE,
    }
    mod = compiler.SourceModule(block_kernel_code)
    matrixmulblock = mod.get_function("MatMulBlockKernel")
    matrixmulblock(a_gpu, b_gpu,c_gpu,grid = (MATRIX_SIZE // TILE_SIZE, MATRIX_SIZE // TILE_SIZE),block = (TILE_SIZE, TILE_SIZE, 1))

    print "Matrix A (GPU):"
    print a_gpu.get()
    print "Matrix B (GPU):"
    print b_gpu.get()
    print "Matrix C (GPU):"
    print c_gpu.get()
    print "CPU-GPU difference:"
    print c_cpu - c_gpu.get()
    print "L2 norm:", la.norm(c_cpu - c_gpu.get())
    
    np.allclose(c_cpu, c_gpu.get())
