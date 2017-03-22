//
//  functions.c
//  
//
//  Created by Eric Dunipace on 3/10/17.
//
//

#include "functions.h"

typedef float floatType_t;

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
#define maxSize = sqrt(16 * pow(2,20)/(3*sizeof(double))
#define NUM_STREAMS 8



__global__ void naive_mm(const int N, const int J, const int K, double *x, double *y, double *z)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(row < N && col < K)
    {
        fot(int k = 0; k < K; k++)
        {
            c[index(row, j, N)] += a[index(row, k, N)] * b[index(k, col, K)];
        }
    }
    return;
}

void cpu_mm(const int N, const int J, const int K, const double *x, const double *y, double *z)
{
    for( int n = 0 ; n < N; ++n )
    {
        for( int j = 0; j < J; j++)
        {
            for( int k = 0; k < K; k++ )
            {
                z[ index(n, k, N) ] += x[index(n, j, N) ] * y[ index(j, k, J) ];
            }
        }
    }
}

int allEqual(const double *a, const double *b, const int N, const double tol)
{
    int equal = 1;
    double diff = 0;
    
    for(int n=0; n < N; ++n){
        diff = fabs(a[n]-b[n]);
        if(diff > tol){
            equal = 0;
            break;
        }
    }
    
    return equal;
}

__global__ void tiled_mm(const int N, const int J, const int K, double *x, double *y, double *z)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    
    
    for( int j = 0; j < n; j++ )
    {
        for( int n = 0; n < N; n++ )
        {
            for( int k = 0; k < K; k++ )
            {
                c[index(n, j, N)] += a[index(n, k, N)] * b[index(k, j, K)];
            }
        }
    }
}

typedef float floatType_t;

/* macro for index calculations */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* matrix size and thread dimensions */

#define SIZE 1024

/* setup various hard-coded parameters for this kernel */

#define TBX 64 // Size of C this CTA is responsible for, x dimension
#define TBY 64 // Size of C this CTA is responsible for, y dimension
#define TX 16 // Thread block size, x dimension
#define TY 16 // Thread block size, y dimension
#define BK 16 // square block of K size
#define NX 4  // = TBX/TX == number of iterations to do TBX work with TX blocks
#define NY 4  // = TBY/TY == number of iterations to do TBY work with TY blocks

__global__ void GPU_shmem2(const int m, floatType_t const * const a,
                           floatType_t const * const b, floatType_t *c )
{
    
    /* setup some constants for later use */
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int iby = blockIdx.y * TBY;
    const int ibx = blockIdx.x * TBX;
    
    /* shared memory arrays for A and B */
    
    __shared__ floatType_t as[ TBX ][ BK+1 ];
    __shared__ floatType_t bs[ BK ][ TBY+1 ];
    
    /* space for C to be held in registers */
    
    floatType_t c_tmp[ NX ][ NY ] ;
    
    /* zero the temp C array */
    
#pragma unroll
    for ( int i = 0 ; i < NX ; i++) {
        for ( int j = 0 ; j < NY ; j++) {
            c_tmp[i][j] = 0.0;
        }
    }
    
    /* calculate my initial offset into A and B */
    
    int aoff = INDX( ibx + tx, ty, m );
    int boff = INDX( tx, iby + ty, m );
    
    /* main loop over blocks of K */
    
    for( int Kblock = 0; Kblock < m; Kblock+=BK )
    {
        
        /* read block of A into shared memory */
        
#pragma unroll
        for ( int i = 0; i < NX ; i ++ )
        {
            as[ tx + i * TX ][ ty ] = a[ (aoff + i*TX) ];
        }
        
        /* read block of B into shared memory */
        
#pragma unroll
        for ( int i = 0; i < NY ; i ++ )
        {
            bs[ tx ][ ty + TY * i ] = b[ (boff + m*i*TY) ];
        }
        
        __syncthreads();
        
        /* increment A and B offsets  for next round of data reads */
        
        boff += BK;
        aoff += m * BK;
        
        /* triply nested loop to perform the matmult on the blocks */
        
#pragma unroll
        for( int k = 0 ; k < BK ; k++ )
        {
#pragma unroll
            for (int j = 0 ; j < NY ; j++ )
            {
#pragma unroll
                for (int i = 0 ; i < NX ; i++ )
                {
                    c_tmp[ i ][ j ] += as[ tx + TX*i ][ k ] * bs[ k ][ ty + j*TY ];
                }
            }
        }
        
        __syncthreads();
        
    } /* end for Kblock */
    
    /* set coff to its proper index int the C matrix */
    
    int coff = INDX( ibx + tx, iby + ty, m );
    
    /* write results to the C matrix */
    
#pragma unroll
    for ( int j = 0 ; j < NY ; j++ ) 
    {
#pragma unroll
        for ( int i = 0 ; i < NX ; i++ )
        {      
            c[ coff + INDX( TX * i, TY * j, m )] = c_tmp[i][j];
        }
    }
    
} /* end GPU_shmem1 */


__global__ void printMat( const floatType_t *A, int size )
{
    if( threadIdx.x == 0 && blockIdx.x == 0 )
        for( int i = 0; i < size; i++ )
            printf("A[%d] = %f\n",i,A[i]);
    return;
} /* end printMat */

void printMatHost( const floatType_t *A, int size )
{
    for( int i = 0; i < size; i++ )
        printf("A[%d] = %f\n",i,A[i]);
    return;
} /* end printMatHost */

