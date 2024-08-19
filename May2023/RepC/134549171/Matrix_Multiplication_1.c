#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
#define ORDER 1000
int A[ORDER][ORDER];
int B[ORDER][ORDER];
int C[ORDER][ORDER];
int main(){
int Ndim, Pdim, Mdim;
int i,j,k,tmp;
double start_time, run_time;
Ndim = ORDER;
Pdim = ORDER;
Mdim = ORDER;
srand(time(NULL));
for (i=0; i<Ndim; i++)
for (j=0; j<Pdim; j++)
A[i][j] = rand() % 100;
for (i=0; i<Pdim; i++)
for (j=0; j<Mdim; j++)
B[i][j] = rand() % 100;
for (i=0; i<Ndim; i++)
for (j=0; j<Mdim; j++)
C[i][j] = 0;
printf("Max number of threads: %i \n",omp_get_max_threads());
start_time = omp_get_wtime();
#pragma omp parallel for private(j,k,tmp) shared(A,B,C)
for (i=0; i<Ndim; i++){
for (j=0; j<Mdim; j++){
tmp = 0;
for(k=0;k<Pdim;k++){
tmp += A[i][k] * B[k][j];
}
C[i][j] = tmp;
}
}
run_time = omp_get_wtime() - start_time;
printf(" Order %d multiplication in %f seconds \n", ORDER, run_time);
return 0;
}
