#!/usr/bin/env python
# -*- coding: utf-8 -*-

####
# copied from: https://andreask.cs.illinois.edu/PyCuda/Examples/MatrixmulSimple
###

import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
    __global__ void MatrixMulKernel(float *a, float *b, float *c)
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


MATRIX_SIZES = [64]

for MATRIX_SIZE in MATRIX_SIZES:

    # initialize matrices
    a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    c_cpu = np.dot(a_cpu, b_cpu)

    # send matrices to GPU
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # compile & call function
    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE
    }
    mod = compiler.SourceModule(kernel_code)
    matrixmul = mod.get_function("MatrixMulKernel")
    matrixmul(a_gpu, b_gpu, c_gpu, block = (MATRIX_SIZE, MATRIX_SIZE, 1))

    # print the results
    print("Matrix A (GPU):")
    #print(a_gpu.get())
    print("Matrix B (GPU):")
    #print(b_gpu.get())
    print("Matrix C (GPU):")
    #print(c_gpu.get())
    print("CPU-GPU difference:")
    print(c_cpu - c_gpu.get())

    np.allclose(c_cpu, c_gpu.get())
