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
double *_P;
_P = (double*) calloc(n * (s+1) * 2, sizeof(double));
if ( _P == NULL ) {
fprintf(stderr, "error: could not allocate P\n");
return 1;
}
double *P[2];
P[0] = &_P[0];
P[1] = &_P[(n+1)*s];
double *mu = (double*) malloc((1 + 2 * s ) * sizeof(double));
if ( mu == NULL ) { 
fprintf(stderr, "error: could not allocate mu\n");
return 1;
}
double *B = (double*) calloc(n * s, sizeof(double));
if ( B == NULL ) { 
fprintf(stderr, "error: could not allocate B\n");
return 1;
}
double *C = (double*) malloc(n * s * sizeof(double));
if ( C == NULL ) { 
fprintf(stderr, "error: could not allocate C\n");
return 1;
}
double *m = (double*) malloc(s * sizeof(double));
if ( m == NULL ) { 
fprintf(stderr, "error: could not allocate m\n");
return 1;
}
double *a = (double*) malloc(s * sizeof(double));
if ( a == NULL ) { 
fprintf(stderr, "error: could not allocate a\n");
return 1;
}
double *accp = (double*) malloc(nb * n * sizeof(double));
if ( accp == NULL ) { 
fprintf(stderr, "error: could not allocate accp\n");
return 1;
}
int IONE = 1; 
double DONE = 1.0, DMONE = -1.0;
double *res = P[1];
mkl_dcsrsymv ("U", &n, mat, vptr, vpos, x, C);  		
dcopy (&n, rhs, &IONE, res, &IONE);                         	
daxpy (&n, &DMONE, C, &IONE, res, &IONE);                    	
struct timeval start;
gettimeofday(&start,NULL);
c_R_mu(s, n, b, mu, R, accp, rhs);
double alpha;
task_ddot_init(b, ar, res, &alpha); 
for ( i=b; i<n-b; i+=b ) {
task_ddot(i, b, n, &ar[i], &res[i], &alpha); 
}
task_ddot_fin(&gamma, b, &ar[ifin], &res[ifin], &alpha);
printf("first alpha %16.20e\n", alpha);
double umbral = 1.0e-8;
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
#pragma omp taskwait
struct timeval stop;
gettimeofday(&stop,NULL);
free(_P);
free(mu);
free(B);
free(C);
free(m);
free(a);
free(accp);
unsigned long elapsed = (stop.tv_sec - start.tv_sec) * 1000000;
elapsed += (stop.tv_usec - start.tv_usec);
return elapsed;
} 
