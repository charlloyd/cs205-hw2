#include <iostream>
#include <stdlib.h>
#include <stdio.h>

int main () {

int b[8] = {1,2,3,4,5,6,7,8};
int total=0;

#pragma acc kernels
#pragma acc data copyin(b), copyout(total)
   for(int a = 0; a < 8; a = a + 1 ) {
      total += b[a];
   }
   printf("result: %d \n",total);
   return 0;
}
