#include <omp.h>
#include <stdio.h>
int tp;
#pragma omp threadprivate(tp)
int var;
int main(){
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
}
return 0;
}
