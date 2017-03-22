#include <iostream>
#include <stdlib>

// compile with "g++ helloacc.cpp -o helloacc"
// compile with "pgc++ -acc helloacc.cpp -Minfo=accel"
// execute with "./helloacc"

using namespace std;

int main () {
   // for loop execution

int b[8] = {1,2,3,4,5,6,7,8};
   
#pragma acc kernels
   for( int a = 0; a < 8; a = a + 1 ) {
      //cout << "Hello World! number: " << b[a] << endl;
      printf("Hello World! number: ",b[a]);
   }
 
   return 0;
}
