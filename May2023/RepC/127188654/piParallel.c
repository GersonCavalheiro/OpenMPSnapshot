#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void Sum(int n, double* sum);
int thread_count=3;
int main(int argc, char* argv[])
{
int n=100000000;
double sum=0.0;
Sum(n,&sum);
printf("for n= %d \n sum: \n ",n,4.0*sum);
return 0;
}
void Sum(int n, double*sum)
{
double factor=1.0;
#pragma omp parallel for num_threads(thread_count) reduction(+:sum) private(factor)
for(int i=0;i<n;i++)
{
*sum +=factor/ (2*i+1);
factor=-factor;
}
}