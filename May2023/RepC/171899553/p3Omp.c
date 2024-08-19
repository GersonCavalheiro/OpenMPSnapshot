#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "timing.h"
int proc;
void Cal(int msize);
int main(int argc, char*argv[])
{
if(argc<3)
{
perror("\n Use the parameter\n");
exit(-1);
}
int matrixSize = atoi(argv[1]);
proc = atoi(argv[2]);
int i, j;
int proc1 = proc;
int rand_num;
srand((unsigned)time(NULL));
rand_num = rand()%10 + 1;
timing_start();
#pragma omp parallel private(i, j) num_threads(proc)
{
int **array = (int**)malloc(sizeof(int *)*matrixSize);
#pragma omp for 
for(i=0; i<matrixSize; ++i)
{
array[i] = (int *)malloc(sizeof(int)*matrixSize);
}
#pragma omp for 
for(i=0; i<matrixSize; ++i)
{
for(j=0; j<matrixSize; ++j)
{
array[i][j] = rand()%10 + 1;
}
}
#pragma omp for schedule(guided, proc)
for(j=0; j<matrixSize; j = j + proc1)
{
array[j] = (int *)malloc(sizeof(int)*matrixSize);
}
for(i=0;i<matrixSize;++i)
free(array[i]);
free(array);
}
timing_stop();
print_timing();
}
