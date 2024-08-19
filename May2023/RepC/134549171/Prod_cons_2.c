
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

#define N        10000


#define SEED       2531
#define RAND_MULT  1366
#define RAND_ADD   150889
#define RAND_MOD   714025
int randy = SEED;


void fill_rand(int length, double *a)
{
int i; 
for (i=0;i<length;i++) {
randy = (RAND_MULT * randy + RAND_ADD) % RAND_MOD;
*(a+i) = ((double) randy)/((double) RAND_MOD);
}   
}


double Sum_array(int length, double *a)
{
int i;  double sum = 0.0;
for (i=0;i<length;i++)  sum += *(a+i);  
return sum; 
}

int main()
{
double *A, sum, runtime;
int flag = 0, flg_temp;

A = (double *)malloc(N*sizeof(double));

runtime = omp_get_wtime();

#pragma omp parallel sections
{
#pragma omp section
{
fill_rand(N, A);        
#pragma omp flush
#pragma omp atomic write
flag = 1;
#pragma omp flush (flag)
}
#pragma omp section
{
#pragma omp flush (flag)
while(1){
#pragma omp flush (flag)
#pragma omp atomic read
flg_temp = flag;
if(flg_temp == 1) break;
}
sum = Sum_array(N, A);  
#pragma omp flush
}
}

runtime = omp_get_wtime() - runtime;

printf(" In %f seconds, The sum is %f \n",runtime,sum);
}

