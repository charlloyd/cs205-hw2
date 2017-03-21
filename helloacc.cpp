#include <iostream>

// compile with "g++ helloacc.cpp -o helloacc"
// execute with "pgcc -acc helloacc.cpp -Minfo=accel"

using namespace std;

int main () {
   // for loop execution
   for( int a = 0; a < 16; a = a + 1 ) {
      cout << "Hello World! number: " << a << endl;
   }
 
   return 0;
}
