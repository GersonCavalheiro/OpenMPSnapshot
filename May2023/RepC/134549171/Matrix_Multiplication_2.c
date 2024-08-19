#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
#define ORDER 1000
int main(){
int Ndim, Pdim, Mdim;
int *A, *B, *C;
int i,j,k,tmp;
double start_time, run_time;
Ndim = ORDER;
Pdim = ORDER;
Mdim = ORDER;
A = (int *)malloc(Ndim*Pdim*sizeof(int));
B = (int *)malloc(Pdim*Mdim*sizeof(int));
C = (int *)malloc(Ndim*Mdim*sizeof(int));
srand(time(NULL));
for (i=0; i<Ndim; i++)
for (j=0; j<Pdim; j++)
*(A+(i*Ndim+j)) = rand() % 100;
for (i=0; i<Pdim; i++)
for (j=0; j<Mdim; j++)
*(B+(i*Pdim+j)) = rand() % 100;
for (i=0; i<Ndim; i++)
for (j=0; j<Mdim; j++)
*(C+(i*Ndim+j)) = 0;
printf("Max number of threads: %i \n",omp_get_max_threads());
start_time = omp_get_wtime();
#pragma omp parallel for private(j,k,tmp) shared(A,B,C)
for (i=0; i<Ndim; i++){
for (j=0; j<Mdim; j++){
tmp = 0;
for(k=0;k<Pdim;k++){
tmp += A[i*Ndim + k] * B[k * Pdim + j];
}
*(C+(i*Ndim+j)) = tmp;
}
}
run_time = omp_get_wtime() - start_time;
printf(" Order %d multiplication in %f seconds \n", ORDER, run_time);
return 0;
}