#include <iostream>
#include <stdlib.h>
#include <stdio.h>

// compile with "g++ helloacc.cpp -o helloacc"
// compile with "pgc++ -acc helloacc.cpp -Minfo=accel"
// execute with "./helloacc"

using namespace std;

int main () {
   // for loop execution

int b[8] = {1,2,3,4,5,6,7,8};
int total=0;
int a = 0;

#pragma acc kernels
#pragma acc data copyin(b), copyout(total), create(a)
   for( a = 0; a < 8; a = a + 1 ) {
      //cout << "Hello World! number: " << b[a] << endl;
      total += b[a];
   }
 
   printf("result: %d \n",total);
   return 0;
}
