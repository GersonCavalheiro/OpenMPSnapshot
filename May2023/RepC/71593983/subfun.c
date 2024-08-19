#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
double f(double x);
void Trap(double a, double b, int n, double * value);
int main()
{
double a, b;
int n = 100;
a = 0.0;
b = 1.0;
double * value = (double *) malloc(sizeof(double));
#pragma omp parallel 
{
Trap(a,b,n,value);	
}
printf("value = %f \n", *value);
return 0;
}
double f(double x)
{
return x;
}
void Trap(double a, double b, int n, double * value)
{
int i;
double h = (b-a)/(double) n;
double val = 0;
int local_n;
double x, local_a, local_b;
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
local_n = n/thread_count;
local_a = a + my_rank*local_n*h;
local_b = local_a + local_n*h;
for(i = 0; i < local_n; i++)
{
x = local_a + i*h;
val += f(x);
}
val *= h;
#pragma omp critical
*value += val;
} 		
