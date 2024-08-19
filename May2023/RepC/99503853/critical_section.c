#include <stdio.h>
#include <omp.h>
int main()
{
int x=0;
#pragma omp parallel num_threads(300)
{
x+=1;
}
printf("Final value of x(Without critical section)is %d\n",x);
x=0;
#pragma omp parallel num_threads(300)
{
#pragma omp critical
{
x+=1;
}
}
printf("Final value of x(With critical section) %d\n",x);
}
