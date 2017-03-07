import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
    __global__ void MatrixMulKernel(float *a, float *b, float *c) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        float Pvalue = 0;

        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
            float Aelement = a[ty * %(MATRIX_SIZE)s + k];
            float Belement = b[k * %(MATRIX_SIZE)s + tx];
            Pvalue += Aelement * Belement; }
        c[ty * %(MATRIX_SIZE)s + tx] = Pvalue; }
""")

MATRIX_SIZE = 2

// create two random square matrices
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

// compute reference on the CPU to verify GPU computation
c_cpu = np.dot(a_cpu, b_cpu)

// transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

// create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

// get the kernel code from the template
// by specifying the constant MATRIX_SIZE
kernel_code = kernel_code_template %
{
    'MATRIX_SIZE': MATRIX_SIZE
}

// get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")

// call the kernel on the card
matrixmul(
    // inputs
    a_gpu, b_gpu,
    // output
    c_gpu,
    // (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (MATRIX_SIZE, MATRIX_SIZE, 1),
    )

// print the results
print "-" * 80
print "Matrix A (GPU):"
print a_gpu.get()

print "-" * 80
print "Matrix B (GPU):"
print b_gpu.get()

print "-" * 80
print "Matrix C (GPU):"
print c_gpu.get()

print "-" * 80
print "CPU-GPU difference:"
print c_cpu - c_gpu.get()

np.allclose(c_cpu, c_gpu.get())
