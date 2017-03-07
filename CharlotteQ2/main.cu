#include <iostream>
#include <stdio.h>

#include "gputimer.h"
void print_array(int *array, int size)
{
printf("{ ");
for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
printf("}\n");
}

// Kernel:
__global__ void square(float* numbers)
{
	// get the array coordinate:
	unsigned int x  = blockIdx.x * blockDim.x + threadIdx.x;
	
	// square the number:
	numbers[x] = numbers[x] * numbers[x];
}

// CPU Code:
int main (int argc, char const* argv[])
{
    GpuTimer timer;

	const unsigned int N = 100;	// N numbers in array

	float data[N];		// array that contains numbers to be squared
	float squared[N];	// array to be filled with squared numbers
	
	// number to be squared will be the index:
	for(unsigned i=0; i<N; i++)
		data[i] = static_cast<float>(i);
	
	// allocate memory on CUDA device:
	float* pDevData;		// pointer to the data on the CUDA Device
	cudaMalloc((void**)&pDevData, sizeof(data));
	
	// copy data to CUDA device:
	cudaMemcpy(pDevData, &data, sizeof(data), cudaMemcpyHostToDevice);
		
	// execute kernel function on GPU:
    timer.Start();
	square<<<10, 10>>>(pDevData);
    timer.Stop();
	
	// copy data back from CUDA Device to 'squared' array:
	cudaMemcpy(&squared, pDevData, sizeof(squared), cudaMemcpyDeviceToHost);
	
	// free memory on the CUDA Device:
	cudaFree(pDevData);
	
	// output results:
	for(unsigned i=0; i<N; i++)
		std::cout<<data[i]<<"^2 = "<<squared[i]<<"\n";

    // timer results:
    printf("Time elapsed = %g ms\n", timer.Elapsed());
	
	return 0;
}
