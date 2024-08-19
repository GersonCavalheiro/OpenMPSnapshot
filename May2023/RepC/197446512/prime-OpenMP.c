#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "myheader.h"
#include <omp.h>

int main(int argc, char** argv){

int n=100000;
int numprimes = 0;
int i;
int tid;
#pragma omp parallel default(shared) private(tid)
{

double start = omp_get_wtime();
tid = omp_get_thread_num();

if (tid==0){
int nthreads = omp_get_num_threads();
printf("Number of threads: %d\n", nthreads);
}

#pragma omp for reduction(+:numprimes)
for (i = 1; i <= n; i++)
{
{
if (is_prime(i) == 1)
numprimes++;
}
}
double end = omp_get_wtime();

printf("Thread ID: %d Time: %f\n",tid,end-start);
if (tid == 0)
printf("Number of Primes: %d\n",numprimes);
}

return 0;

}

int is_prime(int n)
{

if      (n == 0) return 0;
else if (n == 1) return 0;
else if (n == 2) return 1;

int i;
for(i=2;i<=(int)(sqrt((double) n));i++)
if (n%i==0) return 0;

return 1;
}

