#include "helper.cpp"
#include <algorithm>

void apsp(int *A0, int n)
{

   int *Anp1 = (int *)malloc(sizeof(int) * n * n);
   int *An = (int *)malloc(sizeof(int) * n * n);

   copy_from_source_to_destination(A0, Anp1, n*n);
   copy_from_source_to_destination(A0, An  , n*n);

   while(true)
   {
      for(int i=0; i<n; i++)
      {
         for(int j=0; j < n; j++)
         {
           Anp1[i*n+j] = An[i*n+j];

           for(int k=0;k < n; k++) Anp1[i*n+j] = std::min(Anp1[i*n+j], An[i*n+k] + A0[k*n+j]);
         }
      }

      matrix P = matrix(Anp1,n,n);
      P.print();
      getc(stdin);

      if (same(Anp1,An,n*n)) break;
      copy_from_source_to_destination(Anp1, An  , n*n);

   }
}

int main(void)
{
  int data[4] = {2,3,4,1};

  matrix A = matrix(data,2,2);

  A.print();

  apsp(A.data, 2);

}