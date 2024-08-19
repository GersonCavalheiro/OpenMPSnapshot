#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "func.h"

int main(int argc, char *argv[]) {

int max_iter, N,i,j;

if (argc == 3) {
N = atoi(argv[1]) + 2;
max_iter = atoi(argv[2]);
}
else {
N = 128 + 2;
max_iter = 5000;
}
double delta = 2.0/N;

double *f, *u, *u_old;
int size_f = N * N * sizeof(double);
int size_u = N * N * sizeof(double);
int size_u_old = N * N * sizeof(double);

f = (double *)malloc(size_f);
u = (double *)malloc(size_u);
u_old = (double *)malloc(size_u_old);

if (f == NULL || u == NULL || u_old ==NULL) {
fprintf(stderr, "memory allocation failed!\n");
return(1);
}

#pragma omp parallel shared(f,u,u_old,N) private(i,j)
{
#pragma omp for
for (i = 0; i < N; i++){
for (j = 0; j < N; j++){
if (i >= N * 0.5  &&  i <= N * 2.0/3.0  &&  j >= N * 1.0/6.0  &&  j <= N * 1.0/3.0)
f[i*N + j] = 200.0;
else
f[i*N + j] = 0.0; 

if (i == (N - 1) || i == 0 || j == (N - 1)){
u[i*N + j] = 20.0;
u_old[i*N + j] = 20.0;
}
else{
u[i*N + j] = 0.0;
u_old[i*N + j] = 0.0;
} 
}
}

}  

double time_compute = omp_get_wtime();  
jac_cpu(N, delta, max_iter,f,u,u_old);
double tot_time_compute = omp_get_wtime() - time_compute;


double GB = 1.0e-09;
double flop = max_iter * (double)(N-2) * (double)(N-2) * 10.0;
double gflops  = (flop / tot_time_compute) * GB;
double memory  = size_f + size_u + size_u_old;
double memoryGBs  = memory * GB * (1 / tot_time_compute);

printf("%g\t", memory); 
printf("%g\t", gflops); 
printf("%g\t", memoryGBs); 
printf("%g\t", tot_time_compute); 
printf("%g\t", 0); 
printf("%g\t", tot_time_compute); 
printf("# cpu\n");


free(f);
free(u);
free(u_old);
return(0);
}

