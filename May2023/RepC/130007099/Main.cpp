#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define ARRAYSIZE	128 * 1024
#ifndef CHUNKSIZE
#error "Define the chunksize"
#endif
#ifndef NUMT
#error "Define the number of threads"
#endif
float Array[ARRAYSIZE];
float prod;
float Ranf(float, float);
int
main(int argc, char *argv[])
{
#ifndef _OPENMP
fprintf(stderr, "OpenMP is not available\n");
return 1;
#endif
omp_set_num_threads(NUMT);
int numProcessors = omp_get_num_procs();
fprintf(stderr, "Available number of processors: %d \n", numProcessors);
for (int x = 0; x < ARRAYSIZE; x++)
{
Array[x] = Ranf(-1.f, 1.f);
}
printf("Successfully filled the array with random numbers.\n");
double time0 = omp_get_wtime();
int i,j;
#pragma omp parallel for schedule(dynamic, CHUNKSIZE)
for (i = 0; i < ARRAYSIZE; i++)
{
prod = 1.0f;
for (j = 0; j < i + 1; j++)
{
prod *= Array[j];
}
}
double time1 = omp_get_wtime();
long int numMuled = (long int)ARRAYSIZE * (long int)(ARRAYSIZE + 1) / 2;  
fprintf(stderr, "Threads = %2d; ChunkSize = %5d; Scheduling=dynamic ; MegaMults/sec = %10.2lf\n", NUMT, CHUNKSIZE, (double)numMuled / (time1 - time0) / 1000000.);
return 0;
}
float
Ranf(float low, float high)
{
float r = (float)rand();		
return(low + r * (high - low) / (float)RAND_MAX);
}
