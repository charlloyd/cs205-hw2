#include "helper.cpp"

void bfs(int *Adj, int n, int root)
{
   int *x      = (int *) malloc(sizeof(int) * n);
   int *last_x = (int *) malloc(sizeof(int) * n);

   #pragma acc enter data copyin(Adj[0:n * n], x[0:n],last_x[0:n])
   for(int i=0;i < n; i++)
   {
     x[i] = 0; last_x[i] = 0;
   }
   x[root] = 1; last_x[root] = 1;

   while(true)
   {
     copy_from_source_to_destination(x, last_x, n);

    #pragma acc kernels loop independent
     for(int j=0; j < n; j++)
     {
        x[j] = 0;
         
        #pragma acc kernels loop independent
        for(int k=0;k < n; k++)
        {
           x[j] = x[j] || (Adj[j*n+k] && last_x[k]);
        }
     }

     print(x,1,n);
     //print(last_x,1,n);


     if (same(x,last_x,n)) break;

   }
  #pragma acc exit data copyout(x[0:n])
  #pragma acc exit data delete(Adj[0:n * n], last_x[0:n])
}

int main(void)
{
  int data[16]={0,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0};

  bfs(data, 4, 0);
}

