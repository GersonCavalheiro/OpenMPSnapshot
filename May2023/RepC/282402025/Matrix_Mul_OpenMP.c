#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define RANDLIMIT	5	
#define N		1000  	
#define NUMLIMIT 70.0

float a[N][N];
float b[N][N];
float c[N][N];

int main() {
int thread_num = omp_get_max_threads();
int i,j,k;
for (i=0; i<N; i++){
for (j=0; j<N; j++) {
a[i][j] = 1+(int) (NUMLIMIT*rand()/(RAND_MAX+1.0)); 
b[i][j] = (double) (rand() % RANDLIMIT);
}
}
printf("Matrix Size\t  :%d\n",N);
printf("Used Method\t  :OpenMP\n");
printf("Number of Threads :%d\n",thread_num);
clock_t begin = clock();
#pragma omp parallel shared(a,b,c) private(i, j, k)
{
#pragma omp for
for (i=0; i<N; i++) {
for (j=0; j<N; j++) {
c[i][j] = 0.0;
for (k=0; k<N; k++){
c[i][j] = c[i][j] + a[i][k]*b[k][j];
}
}
}
}
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
printf("Time to Multiply  :%.6f Seconds\n",time_spent);
return 0;
}