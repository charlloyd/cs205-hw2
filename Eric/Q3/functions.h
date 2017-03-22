//
//  functions.h
//  
//
//  Created by Eric Dunipace on 3/10/17.
//
//

#ifndef functions_h
#define functions_h

#include <stdio.h>
#include <math.h>
#include </usr/local/cuda-7.0/include/cublas_v2.h>
#include "debug.h"


#define index( row, col, ld ) ( ( (col) * (ld) ) + (row) )

void cpu_mm(const int N, const int J, const int K, const double *x, const double *y, double *z);
int allEqual(const double *a, const double *b, const int N, const double tol);


#endif /* functions_h */
