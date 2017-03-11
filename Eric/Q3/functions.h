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

#define index( row, col, ld ) ( ( (col) * (ld) ) + (row) )

void cpu_mm(const int N, const int J, const int K, const double *x, const double *y, double *z);

#endif /* functions_h */
