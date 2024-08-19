#include <stdio.h>
#include <omp.h>
int main ()
{
omp_set_num_threads(8);
int x=0;
#pragma omp parallel shared(x) 
{
x++;
#pragma omp barrier
}
printf("After first parallel (shared) x is: %d\n", x);
x=0;
#pragma omp parallel private(x)
{
x++;
}
printf("After second parallel (private) x is: %d\n",x);
x=0;
#pragma omp parallel firstprivate(x)
{
x++;
}
printf("After third  parallel (first private) x is: %d\n",x);
return 0;
}
