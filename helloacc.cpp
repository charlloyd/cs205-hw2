#include <iostream>

// compile with "g++ helloacc.cpp -o helloacc"
// compile with "pgc++ -acc helloacc.cpp -Minfo=accel"
// execute with "./helloacc"

using namespace std;

int main () {
   // for loop execution

#pragma acc data create(a)
#pragma acc kernels
   for( int a = 0; a < 16; a = a + 1 ) {
      cout << "Hello World! number: " << a << endl;
   }
 
   return 0;
}
