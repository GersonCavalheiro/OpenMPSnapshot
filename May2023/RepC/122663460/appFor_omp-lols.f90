#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define NUMTHREADS 16 
int numThreads;
double totalTime = 0.0;
FILE* myfile;
#ifdef USE_PAPI
#include <papi.h>  
#endif
#include "vSched.h"
double constraint;
double fs;
#define FORALL_BEGIN(strat, s,e, start, end, tid, numThds )  loop_start_ ## strat (s,e ,&start, &end, tid, numThds);  do {
#define FORALL_END(strat, start, end, tid)  } while( loop_next_ ## strat (&start, &end, tid));
double static_fraction = 1.0;  
int chunk_size  = 10;
#define MAX_ITER 1000 
#define PROBSIZE 16384 
int probSize;
int numIters;
double sum = 0.0;
double globalSum = 0.0;
int iter = 0;
float* a;
float* b;
void dotProdFunc(void* arg)
{
double mySum = 0.0;
int threadNum;
int numThreads;
int i = 0;
while(iter < numIters) 
{
mySum = 0.0;
sum = 0.0;
#pragma omp parallel 
{
threadNum = omp_get_thread_num();
numThreads = omp_get_num_threads();
#ifdef USE_VSCHED
setCDY(static_fraction, constraint, chunk_size); 
int startInd = 0;
int endInd = 0;
FORALL_BEGIN(statdynstaggered, 0, probSize, startInd, endInd, threadNum, numThreads)
for (i = startInd; i < endInd; i++)
#ifdef VERBOSE
if(VERBOSE==1) printf("Thread [%d] : iter = %d \t startInd = %d \t  endInd = %d \t\n", threadNum,iter, startInd, endInd);
#endif
#else
#ifdef VERBOSE
if(VERBOSE==1) printf("Thread [%d] : iter = %d executing a chunk \n", threadNum,iter);
#endif
#pragma omp for schedule(guided, chunk_size)
for (i = 0; i < probSize; i++)
{
#endif
{
mySum += a[i]*b[i];
}
#ifdef USE_VSCHED
FORALL_END(statdynstaggered, startInd, endInd, threadNum)
#else
}
#endif
#pragma omp critical
sum += mySum; 
} 
iter++;
} 
} 
int main(int argc, char* argv[])
{
long i;
double totalTime = 0.0;
int checkSum;
if(argc < 3)
{
printf("Usage: appName [probSize][numIters] (numThreads) (chunksize) (static_fraction) (constraint) \n");
probSize = PROBSIZE;
numIters = MAX_ITER;
}
else
{
probSize = atoi(argv[1]);
numIters = atoi(argv[2]);
}
if(argc > 3) numThreads = atoi(argv[3]);
if(argc > 4) chunk_size = atoi(argv[4]);
if(argc > 5) static_fraction = atof(argv[5]);
if(argc > 6) constraint = atof(argv[6]);
printf("starting OpenMP application using vSched. threads = %d \t probSize = %d \t numIters = %d \n", numThreads, probSize, numIters);
vSched_init(numThreads);
a = (float*)malloc(sizeof(float)*probSize);
b = (float*)malloc(sizeof(float)*probSize);
#pragma omp parallel for
for (int i = 0 ; i < probSize ; i++)
{
a[i] = i*1.0;
b[i] = 1.0;
#ifdef VERBOSE
int myTid = omp_get_thread_num();
printf("tid in init = %d", myTid);
#endif
} 
totalTime = -omp_get_wtime(); 
dotProdFunc(NULL);
totalTime += omp_get_wtime(); 
printf("totalTime: %f \n", totalTime);
myfile = fopen("outFilePerf.dat","a+");
fprintf(myfile, "\t%d\t%d\t%f\t%f\n", numThreads, probSize, static_fraction, totalTime);
fclose(myfile);
printf("Completed the program dot Prod for testing vSched. The solution of the program is: %f \n", sum);
vSched_finalize(numThreads);
}
