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
long cg(SparseMatrix *A, double *rhs, double *x, int s, int b, int maxit) {
int n = A->dim1; 
double *mat = A->vval;
int *vptr = A->vptr;
int *vpos = A->vpos;
int nb = (n + b - 1) / b;
int l = n - (nb-1) * b;
int bfin = (b==n)? 0 : l;
int ifin = (b==n)? 0 : n - l;
double *z = (double*) malloc(sizeof(double) * n);
if ( z == NULL ) {
fprintf(stderr, "error: could not allocate z\n");
return 1;
}
double *ztmp = (double*) calloc(n * nb, sizeof(double));
if ( ztmp == NULL ) { 
fprintf(stderr, "error: could not allocate ytmp\n");
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
struct timeval start;
gettimeofday(&start,NULL);
mkl_dcsrsymv ("U", &n, mat, vptr, vpos, x, z);  		
dcopy (&n, rhs, &IONE, res, &IONE);                            	
daxpy (&n, &DMONE, z, &IONE, res, &IONE);                      	
dcopy (&n, res, &IONE, p, &IONE);                              	
double gamma = ddot (&n, res, &IONE, res, &IONE);              	
double prevgamma;
double tol = sqrt (gamma);                                     	
double umbral = 1.0e-8;
double alpha=1.0;
double beta;
int iter = 0;
while ((iter < maxit) && (tol > umbral)) {
int i; 
for ( i = 0; i < n; i+=b ) {
task_mv(i, b, n, nb, A, p, &z[i], ztmp);
}
for ( i = 0; i < n; i+=b ) {
task_zred(i, b, n, nb, ztmp, &z[i]);
}
task_ddot_init(b, p, z, &alpha); 
for ( i=b; i<n-b; i+=b ) {
task_ddot(i, b, n, &p[i], &z[i], &alpha); 
}
task_ddot_fin(&gamma, bfin, &p[ifin], &z[ifin], &alpha); 
for ( i=0; i<n; i+=b ) {
task_axpy(i, b, n, &alpha, &p[i], &z[i], &x[i], &res[i]); 
}
task_ddot2_init(b, res, &prevgamma, &gamma); 
for ( i=b; i<n-b; i+=b ) {
task_ddot2(b, &res[i], &gamma); 
}
task_ddot2_fin(bfin, &res[ifin], &gamma, &beta, &prevgamma, &tol); 
for ( i=0; i<n; i+=b ) {
task_scalaxpy(i, b, n, &beta, &p[i], &res[i]);
}
iter++;
#pragma omp taskwait on(tol)
#if 0
FILE *fstrm = fopen("naiv.log", "a");
if ( fstrm == NULL ) {
fprintf(stderr, "warning: failed to open %s\n", fstrm);
}
fprintf (fstrm, "iter %i : alpha %e beta %e %.20f gamma %e tol %e\n",iter, alpha, beta, beta, gamma, tol);
fclose(fstrm);
print_dvector ("naiv.log", "x", n, x); 
print_dvector ("naiv.log", "p", n, p); 
#endif
printf ("iter %i tol %20.16e alpha %20.16e beta %20.16e\n",iter, tol, alpha, beta);
}
struct timeval stop;
gettimeofday(&stop,NULL);
printf ("it %i res %e\n", iter, tol);
print_dvector("x.naiv", "x", n, x);
free(z); 
free(ztmp);
free(res);
free(p);
unsigned long elapsed = (stop.tv_sec - start.tv_sec) * 1000000;
elapsed += (stop.tv_usec - start.tv_usec);
return elapsed;
} 
