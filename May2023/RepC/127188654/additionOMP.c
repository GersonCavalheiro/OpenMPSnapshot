#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int add=0,a,b;
int main(int argc, char* argv[])
{
int threads_num=5;
#pragma omp parallel num_threads(threads_num) reduction(+:add)
for(int i=0;i<100;i++)
{
a=1;
b=1;
add+=a+b;
}
printf("sum is= %d \n",add);
return 0;
}
