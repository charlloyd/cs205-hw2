#include <iostream>
#include <stdlib.h>
#include <stdio.h>

int main () {

clock_t begin = clock();
   
int b[8] = {1,2,3,4,5,6,7,8};
int total=0;

#pragma acc data copyin(b), copyout(total)
#pragma acc kernels
   for(int a = 0; a < 8; a = a + 1 ) {
      total += b[a];
   }
   printf("result: %d \n",total);
   
   clock_t end = clock();

   double time_spent = (double)(end-begin) / CLOCKS_PER_SEC;
   
   std::cout<<"time elapsed: "<<time_spent<<std::endl;
   
   return 0;
}


