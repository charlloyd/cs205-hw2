/*
 *  Copyright 2017 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdio.h>
#include <math.h>
#include "../debug.h"

/* definitions of threadblock size in X and Y directions */

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* definition of matrix linear dimension */

#define SIZE 4096

/* macro to index a 1D memory array with 2D indices in column-major order */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for shared memory matrix transpose */

__global__ void smem_cuda_transpose( const int m, 
                                     double const * const a, 
                                     double * const c )
{
	
/* declare a shared memory array */

  __shared__ double smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];
	
/* determine my row and column indices for the error checking code */

  const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
  const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

/* determine my row tile and column tile index */

  const int tileX = blockDim.x * blockIdx.x;
  const int tileY = blockDim.y * blockIdx.y;

  if( myRow < m && myCol < m )
  {
/* read to the shared mem array */
/* HINT: threadIdx.x should appear somewhere in the first argument to */
/* your INDX calculation for both a[] and c[].  This will ensure proper */
/* coalescing. */

   smemArray[threadIdx.x][threadIdx.y] = 
      a[INDX( tileX + threadIdx.x, tileY + threadIdx.y, m )];
  } /* end if */

/* synchronize */
  __syncthreads();
		
  if( myRow < m && myCol < m )
  {
/* write the result */
    c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, m )] = 
           smemArray[threadIdx.y][threadIdx.x];
  } /* end if */
  return;

} /* end smem_cuda_transpose */

void host_transpose( const int m, double const * const a, double * const c )
{
	
/* 
 *  naive matrix transpose goes here.
 */
 
  for( int j = 0; j < m; j++ )
  {
    for( int i = 0; i < m; i++ )
    {
      c[INDX(i,j,m)] = a[INDX(j,i,m)];
    } /* end for i */
  } /* end for j */

} /* end host_dgemm */

int main( int argc, char *argv[] )
{

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

  int size = SIZE;

  fprintf(stdout, "Matrix size is %d\n",size);

/* declaring pointers for array */

  double *h_a, *h_c;
  double *d_a, *d_c;
 
  size_t numbytes = (size_t) size * (size_t) size * sizeof( double );

/* allocating host memory */

  h_a = (double *) malloc( numbytes );
  if( h_a == NULL )
  {
    fprintf(stderr,"Error in host malloc h_a\n");
    return 911;
  }

  h_c = (double *) malloc( numbytes );
  if( h_c == NULL )
  {
    fprintf(stderr,"Error in host malloc h_c\n");
    return 911;
  }

/* allocating device memory */

  checkCUDA( cudaMalloc( (void**) &d_a, numbytes ) );
  checkCUDA( cudaMalloc( (void**) &d_c, numbytes ) );

/* set result matrices to zero */

  memset( h_c, 0, numbytes );
  checkCUDA( cudaMemset( d_c, 0, numbytes ) );

  fprintf( stdout, "Total memory required per matrix is %lf MB\n", 
     (double) numbytes / 1000000.0 );

/* initialize input matrix with random value */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
  } /* end for */

/* copy input matrix from host to device */

  checkCUDA( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );

/* create and start timer */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call naive cpu transpose function */

  host_transpose( size, h_a, h_c );

/* stop CPU timer */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print CPU timing information */

  fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* setup threadblock size and grid sizes */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( ( size / THREADS_PER_BLOCK_X ) + 1, 
               ( size / THREADS_PER_BLOCK_Y ) + 1, 1 );

/* start timers */
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call smem GPU transpose kernel */

  smem_cuda_transpose<<< blocks, threads >>>( size, d_a, d_c );
  checkKERNEL()

/* stop the timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print GPU timing information */

  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* copy data from device to host */

  checkCUDA( cudaMemset( d_a, 0, numbytes ) );
  checkCUDA( cudaMemcpy( h_a, d_c, numbytes, cudaMemcpyDeviceToHost ) );

/* compare GPU to CPU for correctness */

  int success = 1;
  
  for( int j = 0; j < size; j++ )
  {
    for( int i = 0; i < size; i++ )
    {
      if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
      {
        printf("Error in element %d,%d\n", i,j );
        printf("Host %f, device %f\n",h_c[INDX(i,j,size)],
                                      h_a[INDX(i,j,size)]);
        success = 0;
        break;
      }
    } /* end for i */
  } /* end for j */

  if( success == 1 ) printf("PASS\n");
  else               printf("FAIL\n");

/* free the memory */

  free( h_a );
  free( h_c );
  checkCUDA( cudaFree( d_a ) );
  checkCUDA( cudaFree( d_c ) );

  checkCUDA( cudaDeviceReset() );

  return 0;
}
