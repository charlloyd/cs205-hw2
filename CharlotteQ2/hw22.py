import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

print("working")

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

print("still working...")

MATRIX_SIZE = 2

a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
c_cpu = np.dot(a_cpu, b_cpu)

a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# compile & call function
matrixmul = mod.get_function("MatrixMulKernel")
print("compiled")
matrixmul(a_gpu, b_gpu, c_gpu, block = (MATRIX_SIZE, MATRIX_SIZE, 1))
print("called")

# print the results
print("Matrix A (GPU):")
print(a_gpu.get())
print("Matrix B (GPU):")
print(b_gpu.get())
print("Matrix C (GPU):")
print(c_gpu.get())
print("CPU-GPU difference:")
print(c_cpu - c_gpu.get())

np.allclose(c_cpu, c_gpu.get())
