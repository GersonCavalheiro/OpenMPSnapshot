#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include<time.h>
static long iterations = 1000000;
int main(int argc, char* argv[])
{
double x, y, z;                  
long count = 0;                     
double pi;                       
int numthreads = 3;
long i;
double time;
time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
printf(" %d threads ",omp_get_num_threads());
srand48((int)time(NULL) ^ omp_get_thread_num());  
#pragma omp for reduction(+:count) private(x,y,z)
for (i = 0; i<iterations; ++i)         
{
x = (double)drand48();
y = (double)drand48();
z = ((x * x) + (y * y));
if (z <= 1)
++count;
}
} 
pi = ((double)count / (double)(iterations)) * 4.0;
printf("Estimated Pi: %f\n", pi);
printf("Time: %f", omp_get_wtime()-time);
return 0;
}
