#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
void fill( double* m, int n )
{
int i, j;
for (i=0; i<n; i++) {
for (j=0; j<n; j++) {
m[i*n + j] = (double)rand() / RAND_MAX;
}
}
}
int min(int a, int b)
{
return (a < b ? a : b);
}
void matmul( double *p, double* q, double *r, int n)
{
int i, j, k;
#pragma omp parallel for collapse(2) private(k)
for (i=0; i<n; i++) {
for (j=0; j<n; j++) {
double s = 0.0;
for (k=0; k<n; k++) {
s += p[i*n + k] * q[k*n + j];
}
r[i*n + j] = s;
}
}
}
void matmul_transpose( double *p, double* q, double *r, int n)
{
int i, j, k;
double *qT = (double*)malloc( n * n * sizeof(double) );
#pragma omp parallel
{
#pragma omp for collapse(2)
for (i=0; i<n; i++) {
for (j=0; j<n; j++) {
qT[j*n + i] = q[i*n + j];
}
}    
#pragma omp for collapse(2) private(k)
for (i=0; i<n; i++) {
for (j=0; j<n; j++) {
double s = 0.0;
for (k=0; k<n; k++) {
s += p[i*n + k] * qT[j*n + k];
}
r[i*n + j] = s;
}
}
}
free(qT);
}
int main( int argc, char *argv[] )
{
int n = 1000;
double *p, *q, *r;
double tstart, tstop;
if ( argc > 2 ) {
printf("Usage: %s [n]\n", argv[0]);
return EXIT_FAILURE;
}
if ( argc == 2 ) {
n = atoi(argv[1]);
}
p = (double*)malloc( n * n * sizeof(double) );
q = (double*)malloc( n * n * sizeof(double) );
r = (double*)malloc( n * n * sizeof(double) );
fill(p, n);
fill(q, n);
tstart = hpc_gettime();
matmul_transpose(p, q, r, n);
tstop = hpc_gettime();
printf("Execution time %f\n", tstop - tstart);  
free(p);
free(q);
free(r);
return EXIT_SUCCESS;
}
