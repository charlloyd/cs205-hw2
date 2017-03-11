//
//  functions.c
//  
//
//  Created by Eric Dunipace on 3/10/17.
//
//

#include "functions.h"

//__global__ void naive_mm(const int N, const int J, const int K, double *x, double *y, double *z)
//{
//    int row = blockDim.x * blockIdx.x + threadIdx.x;
//    int col = blockDim.y * blockIdx.y + threadIdx.y;
//    
//    if()
//    for( int j = 0; j < n; j++ )
//    {
//        for( int i = 0; i < m; i++ )
//        {
//            for( int k = 0; koff < k; koff++ )
//            {
//                c[index(i, j, m)] += a[index( i, koff, m )] * b[index( koff, j, n )];
//            } /* end for i */
//        } /* end jb */
//    }
//    return;
//}

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

