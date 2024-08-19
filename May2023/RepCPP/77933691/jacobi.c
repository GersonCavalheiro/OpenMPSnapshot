#include <stdio.h>

void
jacobi(double * unew, double * uold, double * f,
double lambda, int N, int kmax, int * k){

double lambda2 = lambda*lambda;
int M = N+2;
double* swapper;
*k = 0;

#pragma omp parallel shared(k, unew, uold, f, lambda2, M)
{
while (*k < kmax) {

#pragma omp for
for (int i = 1; i < N+1; i++) {
for (int j = 1; j < N+1; j++) {
unew[i*M+j] = ( 0.25*(uold[(i-1)*M+j]+uold[(i+1)*M+j]+
uold[i*M+j-1]+uold[i*M+j+1]+lambda2*f[i*M+j]) );
}
} 
#pragma omp single
{
(*k)++;
swapper = uold;
uold = unew;
unew = swapper;
} 
}
}
swapper = unew;
unew = uold;
unew = swapper;
}
