#include <iostream>
#include <random>

// compile with "g++ helloworld.cpp -o helloworld"
// execute with "./helloworld"

using namespace std;

int main () {
   // for loop execution
   for( int a = 0; a < 16; a = a + 1 ) {
      cout << "Hello World! number: " << a << endl;
   }
 
   return 0;
}
