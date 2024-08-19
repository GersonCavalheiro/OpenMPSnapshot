#include <stdio.h>
#include <omp.h>
#define SIZE 100
int main ( int argc, char* argv[] )
{
int i, error = 0, step = 1;
int A[SIZE];
for (i = 0; i<SIZE; i++) A[i]=0;
#pragma omp for schedule(ompss_dynamic)
for (i = 0; i<SIZE; i+=1)
A[i]++;
#pragma omp for schedule(ompss_dynamic)
for (i = 0; i<SIZE; i+=step)
A[i]--;
for (i = 0; i<SIZE; i++) if (A[i]!= 0) error++;
fprintf(stdout,"Result with %d threads is %s\n",
omp_get_num_threads(),
error?"UNSUCCESSFUL":"Successful");
return error;
}
