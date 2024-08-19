#include <stdio.h>
#include <stdlib.h>
#include "func.h"
#include <omp.h>


int main(int argc, char *argv[]) {

double ts, te, d, mflops, memory,flop;
int max_iter, loops, N, n_cores;

loops = 1;

double   *f;
double   *u;
double   *u_old;

if (argc == 4 ) {
N = atoi(argv[1]) + 2;
max_iter = atoi(argv[2]);
d = atoi(argv[3]);
}
else {
N = 32 + 2;
max_iter = 100;
d = 0.001;
}

f = (double *)malloc( N * N * sizeof(double));
u = (double *)malloc( N * N * sizeof(double));
u_old = (double *)malloc( N * N * sizeof(double));

if (f == NULL || u == NULL || u_old ==NULL) {
fprintf(stderr, "memory allocation failed!\n");
return(1);
}

double delta = 2.0/N;

int i,j;
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

ts = omp_get_wtime();
for(int i = 0; i < loops; i++ ) {
n_cores = jac_mp_v3(N, delta, d, max_iter,f,u,u_old);
}
te = omp_get_wtime() - ts;

flop=max_iter * (double)(N-2) * (double)(N-2) * 10.0;

mflops  = flop * 1.0e-06 * loops / te;
memory  = 3.0 * (double)(N-2) * (double)(N-2) * sizeof(double);

printf("%d\t", n_cores);
printf("%g\t", memory);
printf("%g\t", mflops);
printf("%g\n", te / loops);

free(f);
free(u);
free(u_old);
return(0);
}

