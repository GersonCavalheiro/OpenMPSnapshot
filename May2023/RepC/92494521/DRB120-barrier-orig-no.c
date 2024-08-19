#include <omp.h>
#include <stdio.h>
int main(int argc, char* argv[])
{
int var = 0;
#pragma omp parallel shared(var)
{
#pragma omp single
var++;
#pragma omp barrier
#pragma omp single
var++;
}
if(var != 2) printf("%d\n",var);
int error = (var != 2);
return error;
}
