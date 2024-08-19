#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
int tp;
#pragma omp threadprivate(tp)
int var;
void work()
{
#pragma omp task
{
#pragma omp task
{
tp = 1;
#pragma omp task
{
}
var = tp; 
}
tp = 2;
}
}
int main()
{
work();
printf("VALUE OF VAR IS : %d\n",var);
return 0;
}
