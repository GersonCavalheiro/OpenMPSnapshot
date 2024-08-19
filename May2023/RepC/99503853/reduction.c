#include<stdio.h>
#include<omp.h>
int main()
{
int i,add=0;
#pragma omp parallel
{
#pragma omp for reduction(+:add) ordered
for(i=1;i<5;i++)
{
#pragma omp ordered
add=i*i;
printf("add=%d by thread=%d\n",add,omp_get_thread_num());
}
}
printf("add=%d\n",add);
}
