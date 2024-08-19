#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mkl.h"
#include "mkl_spblas.h"
#include "cg_main.h"
#include "vector.h"
#include "sparsematrix.h"
#include "finout.h"
#include "cg_kernels.h"
#include "cg_kernels_lookahead.h"
#include "cg_kernels_lookahead2.h"
long cg(SparseMatrix *A, double *rhs, double *x, int s, int b, int maxit) {
int n = A->dim1; 
double *mat = A->vval;
int *vptr = A->vptr;
int *vpos = A->vpos;
int nb = (n + b - 1) / b;
int l = n - (nb-1) * b;
int bfin = (b==n)? 0 : l;
int ifin = (b==n)? 0 : n - l;
double *ar = (double*) malloc(sizeof(double) * n);
if ( ar == NULL ) {
fprintf(stderr, "error: could not allocate ar\n");
return 1;
}
double *ap = (double*) calloc(n, sizeof(double));
if ( ap == NULL ) {
fprintf(stderr, "error: could not allocate ap\n");
return 1;
}
double *z = (double*) calloc(n * nb, sizeof(double));
if ( z == NULL ) { 
fprintf(stderr, "error: could not allocate z\n");
return 1;
}
double *artmp = (double*) calloc(n * nb, sizeof(double));
if ( artmp == NULL ) { 
fprintf(stderr, "error: could not allocate artmp\n");
return 1;
}
double *res = (double*) malloc(sizeof(double) * n);
if ( res == NULL ) {
fprintf(stderr, "error: could not allocate res\n");
return 1;
}
double *p = (double*) malloc(sizeof(double) * n);
if ( p == NULL ) {
fprintf(stderr, "error: could not allocate p\n");
return 1;
}
int IONE = 1; 
double DONE = 1.0, DMONE = -1.0;
mkl_dcsrsymv ("U", &n, mat, vptr, vpos, x, z);  		
dcopy (&n, rhs, &IONE, res, &IONE);                            	
daxpy (&n, &DMONE, z, &IONE, res, &IONE);                      	
dcopy (&n, res, &IONE, p, &IONE);                              	
double gamma = ddot (&n, res, &IONE, res, &IONE);              	
double gamma2 = 0;
double tol = sqrt (gamma);                                     	
struct timeval start;
gettimeofday(&start,NULL);
int i;
for ( i = 0; i < n; i+=b ) {
task_mv(i, b, n, nb, A, res, &ar[i], artmp);
}
for ( i = 0; i < n; i+=b ) {
task_zred(i, b, n, nb, artmp, &ar[i]);
}
double alpha;
task_ddot_init(b, ar, res, &alpha); 
for ( i=b; i<n-b; i+=b ) {
task_ddot(i, b, n, &ar[i], &res[i], &alpha); 
}
task_ddot_fin(&gamma, b, &ar[ifin], &res[ifin], &alpha);
printf("first alpha %16.20e\n", alpha);
double umbral = 1.0e-8;
double beta=0;
double arr;
int iter = 0;
while ((iter < maxit) && (tol > umbral)) {
int i; 
for ( i=0; i<n; i+=b ) {
lh2task_axpy(i, b, n, &alpha, &beta, &p[i], &ap[i], &x[i], &res[i], &ar[i]);
}
for ( i = 0; i < n; i+=b ) {
task_mv(i, b, n, nb, A, res, &ar[i], artmp);
}
for ( i = 0; i < n; i+=b ) {
task_zred(i, b, n, nb, artmp, &ar[i]);
}
lh2task_ddot_init(b, res, ar, &gamma2, &arr); 
for ( i=b; i<n-b; i+=b ) {
lh2task_ddot(b, &res[i], &ar[i], &gamma2, &arr); 
}
lh2task_ddot_fin(bfin, &res[ifin], &ar[ifin], &gamma2, &arr, &alpha, &beta, &gamma); 
iter++;
#pragma omp taskwait on(gamma)
#pragma omp taskwait 
tol = sqrt (gamma);      
#if 0
FILE *fstrm = fopen("lookahead.log", "a");
if ( fstrm == NULL ) {
fprintf(stderr, "warning: failed to open %s\n", fstrm);
}
#endif
printf ("iter %i : alpha %16.20e beta %16.20e gamma %e tol %16.20e\n",iter, alpha, beta, gamma, tol);
#if 0
fprintf (fstrm, "iter %i : alpha %e beta %e %.20f gamma %e tol %e\n",iter, alpha, beta, beta, gamma, tol);
fclose(fstrm);
print_dvector ("lookahead.log", "x", n, x); 
print_dvector ("lookahead.log", "p", n, p); 
#endif
}
struct timeval stop;
gettimeofday(&stop,NULL);
free(ar);
free(ap);
free(z);
free(artmp);
free(res);
free(p);
unsigned long elapsed = (stop.tv_sec - start.tv_sec) * 1000000;
elapsed += (stop.tv_usec - start.tv_usec);
return elapsed;
} 
