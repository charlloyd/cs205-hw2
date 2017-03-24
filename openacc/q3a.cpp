#include "helper.cpp"

void bfs(int *Adj, int n, int root)
{
   int *x      = (int *) malloc(sizeof(int) * n);
   int *last_x = (int *) malloc(sizeof(int) * n);

   for(int i=0;i < n; i++)
   {
     x[i] = 0; last_x[i] = 0;
   }
   x[root] = 1; last_x[root] = 1;

   while(true)
   {
     copy_from_source_to_destination(x, last_x, n);


     for(int j=0; j < n; j++)
     {
        x[j] = 0;

        for(int k=0;k < n; k++)
        {
           x[j] = x[j] || (Adj[j*n+k] && last_x[j]);
        }
     }

     std::cout<<std::endl;
     for(int ii=0;ii<n;++ii) std::cout<<x[ii]<<" "; std::cout<<std::endl;
     for(int ii=0;ii<n;++ii) std::cout<<last_x[ii]<<" "; std::cout<<std::endl;


     if (same(x,last_x,n)) break;

   }
}

int main(void)
{
  int data[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};

  bfs(data, 4, 1);
}

