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
#include "cublas_v2.h"
#include "../debug.h"

typedef float floatType_t;

/* macro for index calculations */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* matrix size and thread dimensions */

/* define blocksize X and blocksize Y and blocksize K */

#define THREADS_PER_BLOCK_X 64 // Thread block size, x dimension
#define THREADS_PER_BLOCK_Y 64 // Thread block size, y dimension
#define BLOCK_K 64 // square block of K size

__global__ void GPU_shmem2(const int m, floatType_t const * const a, 
      floatType_t const * const b, floatType_t *c )
{

/* setup some constanst for later use */

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int iby = blockIdx.y * THREADS_PER_BLOCK_Y;
  const int ibx = blockIdx.x * THREADS_PER_BLOCK_X;

/* shared memory arrays for A and B */

  __shared__ floatType_t as[ THREADS_PER_BLOCK_X ][ BLOCK_K + 1 ];
  __shared__ floatType_t bs[ BLOCK_K ][ THREADS_PER_BLOCK_Y + 1 ];
	
/* space for C to be held in registers */

  floatType_t c_tmp = 0.0 ;

/* calculate my initial offset into A and B */

  int aoff = INDX( ibx + tx, ty, m );
  int boff = INDX( tx, iby + ty, m );

/* main loop over blocks of K */

  for( int Kblock = 0; Kblock < m; Kblock+=BLOCK_K )
  {

/* read block of A into shared memory */
    as[ tx ][ ty ] = a[ aoff ];

/* read block of B into shared memory */
    bs[ tx ][ ty ] = b[ boff ];
		
    __syncthreads();

/* increment A and B offsets  for next round of data reads */
    boff += BLOCK_K;
    aoff += m * BLOCK_K;

/* triply nested loop to perform the matmult on the blocks */

#pragma unroll
    for( int k = 0 ; k < BLOCK_K ; k++ )
    {
      c_tmp += as[ tx ][ k ] * bs[ k ][ ty ];
    }

    __syncthreads();

  } /* end for Kblock */

/* set C to its proper index int the C matrix */
  int coff = INDX( ibx + tx, iby + ty, m );

/* write results to the C matrix */
  c[ coff ] = c_tmp;
 
} /* end GPU_shmem2 */


int main( int argc, char *argv[] )
{

/* get GPU device number and name */
const int SIZE[1] =  {pow(2,10)};//, pow(2,16)};
for(size_t s = 0; s < 1; s++){
  const int size = SIZE[s];
  int dev;
  cudaDeviceProp deviceProp;
  checkCUDA( cudaGetDevice( &dev ) );
  checkCUDA( cudaGetDeviceProperties( &deviceProp, dev ) );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );


  fprintf(stdout, "Matrix size is %d\n",size);

  floatType_t *h_a, *h_b, *h_c, *h_c1;
  floatType_t *d_a, *d_b, *d_c;
 
  size_t numbytes = (size_t ) size * (size_t ) size * sizeof( floatType_t );

  h_a = (floatType_t *) malloc( numbytes );
  if( h_a == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_b = (floatType_t *) malloc( numbytes );
  if( h_b == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_c = (floatType_t *) malloc( numbytes );
  if( h_c == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

  h_c1 = (floatType_t *) malloc( numbytes );
  if( h_c1 == NULL )
  {
    fprintf(stderr,"Error in host malloc\n");
    return 911;
  }

/* zero out the host memory for C matrices */

  memset( h_c, 0, numbytes );
  memset( h_c1, 0, numbytes );

  fprintf( stdout, "Total memory required is %lf MB\n", 
     3.0 * (double) numbytes / 1000000.0 );

/* initialize the A and B matrices */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
    h_b[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
  }

/* allocate a, b, c in gpu memory */

  checkCUDA( cudaMalloc( (void **)&d_a, numbytes ) );
  checkCUDA( cudaMalloc( (void **)&d_b, numbytes ) );
  checkCUDA( cudaMalloc( (void **)&d_c, numbytes ));
	
/* copy a and b to device */

  checkCUDA( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );
  checkCUDA( cudaMemcpy( d_b, h_b, numbytes, cudaMemcpyHostToDevice ) );

  cublasHandle_t handle;
  checkCUBLAS( cublasCreate( &handle ) );

  floatType_t alpha = 1.0;
  floatType_t beta  = 0.0;

/* start timers */

  cudaEvent_t start, stop;
  checkCUDA( cudaEventCreate( &start ) );
  checkCUDA( cudaEventCreate( &stop ) );
  checkCUDA( cudaEventRecord( start, 0 ) );

/* call CUBLAS dgemm */

  if( sizeof( floatType_t ) == 4 )
  {
  checkCUBLAS(
  cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
               size, size, size,
               (float *)&alpha, 
               (float *)d_a, size,
               (float *)d_b, size,
               (float *)&beta,
               (float *)d_c, size )
             );
  } /* end if */
  else
  {
  checkCUBLAS(
  cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
               size, size, size,
               (double *)&alpha, 
               (double *)d_a, size,
               (double *)d_b, size,
               (double *)&beta,
               (double *)d_c, size )
             );
  } /* end else */

/* stop timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  float elapsedTime;
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print GPU CUBLAS timing information */

  fprintf(stdout, "Total time GPU CUBLAS is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GFlop/s\n", 
    2.0 * (double) size * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
    
/* copy C from device to host for error checking */

  checkCUDA( cudaMemcpy( h_c, d_c, numbytes, cudaMemcpyDeviceToHost ) );

/* reset C on device to zero */

  checkCUDA( cudaMemset( d_c, 0, numbytes ) );

/* setup grid and block sizes */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( size / THREADS_PER_BLOCK_X, size / THREADS_PER_BLOCK_Y, 1 );

/* start timers */

  checkCUDA( cudaEventRecord( start, 0 ) );

/* call GPU_naive */

  GPU_shmem2<<< blocks, threads >>> ( size, d_a, d_b, d_c );
  checkKERNEL()

/* stop timers */

  checkCUDA( cudaEventRecord( stop, 0 ) );
  checkCUDA( cudaEventSynchronize( stop ) );
  checkCUDA( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* print data for GPU naive */

  fprintf(stdout, "Total time GPU SHMEM is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GFlop/s\n", 
    2.0 * (double) size * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
                  
/* copy C back to host */
	
  checkCUDA( cudaMemcpy( h_c1, d_c, numbytes, cudaMemcpyDeviceToHost ) );

  checkCUBLAS( cublasDestroy( handle ) );
  checkCUDA( cudaEventDestroy( start ) );
  checkCUDA( cudaEventDestroy( stop ) );

/* check CUBLAS versus GPU NAIVE numerical results */

  double temp = 0.0;

  for( int i = 0; i < size * size; i++ )
  {
     temp = max( temp, abs( (double)h_c[i] - (double)h_c1[i] )/
                      abs((double)h_c[i]) );
  } /* end for */
  printf("Maximum error is %e percent \n",temp*100.0);
  if( temp > 0.001 ) printf("FAIL\n");
  else printf("PASS\n");

/* cleanup */

  checkCUDA( cudaFree( d_a ) );
  checkCUDA( cudaFree( d_b ) );
  checkCUDA( cudaFree( d_c ) );

  free( h_a );
  free( h_b );
  free( h_c );
  free( h_c1 );

  checkCUDA( cudaDeviceReset() );
}
  return 0;
}
