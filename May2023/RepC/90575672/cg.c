#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NZ+1+NA];
static int arow[NA+1];
static int acol[NAZ];
static double v[NZ];
static double aelt[NAZ];
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];
static int myid, num_threads, ilow, ihigh;
#pragma omp threadprivate(myid, num_threads, ilow, ihigh)
#define max_threads 1024
static int last_n[max_threads+1];
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
#pragma omp threadprivate (amult,tran)
static logical timeron;
static void conj_grad(int colidx[],
int rowstr[],
double x[],
double z[],
double a[],
double p[],
double q[],
double r[],
double *rnorm);
static void makea(int n,
int nz,
double a[],
int colidx[],
int rowstr[],
int firstrow,
int lastrow,
int firstcol,
int lastcol,
int arow[],
int acol[][NONZER+1],
double aelt[][NONZER+1],
double v[],
int iv[]);
static void sparse(double a[],
int colidx[],
int rowstr[],
int n,
int nz,
int nozer,
int arow[],
int acol[][NONZER+1],
double aelt[][NONZER+1],
int firstrow,
int lastrow,
int last_n[],
double v[],
int iv[],
int nzloc[],
double rcond,
double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
int main(int argc, char *argv[])
{
int i, j, k, it;
double zeta;
double rnorm;
double norm_temp1, norm_temp2;
double t, mflops, tmax;
char Class;
logical verified;
double zeta_verify_value, epsilon, err;
char *t_names[T_last];
for (i = 0; i < T_last; i++) {
timer_clear(i);
}
FILE *fp;
if ((fp = fopen("timer.flag", "r")) != NULL) {
timeron = true;
t_names[T_init] = "init";
t_names[T_bench] = "benchmk";
t_names[T_conj_grad] = "conjgd";
fclose(fp);
} else {
timeron = false;
}
timer_start(T_init);
firstrow = 0;
lastrow  = NA-1;
firstcol = 0;
lastcol  = NA-1;
if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10) {
Class = 'S';
zeta_verify_value = 8.5971775078648;
} else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12) {
Class = 'W';
zeta_verify_value = 10.362595087124;
} else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20) {
Class = 'A';
zeta_verify_value = 17.130235054029;
} else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60) {
Class = 'B';
zeta_verify_value = 22.712745482631;
} else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110) {
Class = 'C';
zeta_verify_value = 28.973605592845;
} else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500) {
Class = 'D';
zeta_verify_value = 52.514532105794;
} else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500) {
Class = 'E';
zeta_verify_value = 77.522164599383;
} else {
Class = 'U';
}
printf("\n\n NAS Parallel Benchmarks (NPB3.3-OMP-C) - CG Benchmark\n\n");
printf(" Size: %11d\n", NA);
printf(" Iterations:                  %5d\n", NITER);
printf(" Number of available threads: %5d\n", omp_get_max_threads());
printf("\n");
naa = NA;
nzz = NZ;
tran    = 314159265.0;
amult   = 1220703125.0;
zeta    = randlc(&tran, amult);
makea(naa, nzz, a, colidx, rowstr, 
firstrow, lastrow, firstcol, lastcol, 
arow, 
(int (*)[NONZER+1])(void*)acol, 
(double (*)[NONZER+1])(void*)aelt,
v, iv);
#pragma omp barrier
#pragma omp parallel private(it,i,j,k)
{   
#pragma omp for nowait
for (j = 0; j < lastrow - firstrow + 1; j++) {
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
colidx[k] = colidx[k] - firstcol;
}
}
#pragma omp for nowait
for (i = 0; i < NA+1; i++) {
x[i] = 1.0;
}
#pragma omp for
for (j = 0; j < lastcol - firstcol + 1; j++) {
q[j] = 0.0;
z[j] = 0.0;
r[j] = 0.0;
p[j] = 0.0;
}
#pragma omp single
zeta = 0.0;
for (it = 1; it <= 1; it++) {
conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
/ \
reduction(+:norm_temp1,norm_temp2)
for (j = 0; j < lastcol - firstcol + 1; j++) {
norm_temp1 = norm_temp1 + x[j]*z[j];
norm_temp2 = norm_temp2 + z[j]*z[j];
}
#pragma omp single
{ 
norm_temp2 = 1.0 / sqrt(norm_temp2);
zeta = SHIFT + 1.0 / norm_temp1;
}
if (it == 1) 
printf("\n   iteration           ||r||                 zeta\n");
printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);
#pragma omp parallel for
for (j = 0; j < lastcol - firstcol + 1; j++) {
x[j] = norm_temp2 * z[j];
}
} 
#if defined(_OPENMP)
#pragma omp master
#endif 
timer_stop(T_bench);
t = timer_read(T_bench);
printf(" Benchmark completed\n");
epsilon = 1.0e-10;
if (Class != 'U') {
err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
if (err <= epsilon) {
verified = true;
printf(" VERIFICATION SUCCESSFUL\n");
printf(" Zeta is    %20.13E\n", zeta);
printf(" Error is   %20.13E\n", err);
} else {
verified = false;
printf(" VERIFICATION FAILED\n");
printf(" Zeta                %20.13E\n", zeta);
printf(" The correct zeta is %20.13E\n", zeta_verify_value);
}
} else {
verified = false;
printf(" Problem size unknown\n");
printf(" NO VERIFICATION PERFORMED\n");
}
if (t != 0.0) {
mflops = (double)(2*NITER*NA)
* (3.0+(double)(NONZER*(NONZER+1))
+ 25.0*(5.0+(double)(NONZER*(NONZER+1)))
+ 3.0) / t / 1000000.0;
} else {
mflops = 0.0;
}
print_results("CG", Class, NA, 0, 0,
NITER, t,
mflops, "          floating point", 
verified, NPBVERSION, COMPILETIME,
CS1, CS2, CS3, CS4, CS5, CS6, CS7);
if (timeron) {
tmax = timer_read(T_bench);
if (tmax == 0.0) tmax = 1.0;
printf("  SECTION   Time (secs)\n");
for (i = 0; i < T_last; i++) {
t = timer_read(i);
if (i == T_init) {
printf("  %8s:%9.3f\n", t_names[i], t);
} else {
printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
if (i == T_conj_grad) {
t = tmax - t;
printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t*100.0/tmax);
}
}
}
}
return 0;
}
static void conj_grad(int colidx[],
int rowstr[],
double x[],
double z[],
double a[],
double p[],
double q[],
double r[],
double *rnorm)
{
int j, k;
int cgit, cgitmax = 25;
double d, sum, rho, rho0, alpha, beta, suml;
/
#pragma omp for reduction(+:d)
for (j = 0; j < lastcol - firstcol + 1; j++) {
d = d + p[j]*q[j];
}
alpha = rho0 / d;
#pragma omp for 
for (j = 0; j < lastcol - firstcol + 1; j++) {
z[j] = z[j] + alpha*p[j];
r[j] = r[j] - alpha*q[j];
rho = rho + r[j]*r[j];
}
beta = rho / rho0;
#pragma omp for
for (j = 0; j < lastcol - firstcol + 1; j++) {
p[j] = r[j] + beta*p[j];
}
} 
#pragma omp for
for (j = 0; j < lastrow - firstrow + 1; j++) {
suml = 0.0;
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
suml = suml + a[k]*z[colidx[k]];
}
r[j] = suml;
}
#pragma omp for reduction(+:sum) nowait
for (j = 0; j < lastcol-firstcol+1; j++) {
suml = x[j] - r[j];
sum  = sum + suml*suml;
}
}
#pragma omp single
{
*rnorm = sqrt(sum);
}
}
static void makea(int n,
int nz,
double a[],
int colidx[],
int rowstr[],
int firstrow,
int lastrow,
int firstcol,
int lastcol,
int arow[],
int acol[][NONZER+1],
double aelt[][NONZER+1],
double v[],
int iv[])
{
int iouter, ivelt, nzv, nn1;
int ivc[NONZER+1];
double vc[NONZER+1];
int work; 
nn1 = 1;
do {
nn1 = 2 * nn1;
} while (nn1 < n);
num_threads = omp_get_num_threads();
myid = omp_get_thread_num();
if (num_threads > max_threads) {
if (myid == 0) {
printf(" Warning: num_threads%6d exceeded an internal limit%6d\n",
num_threads, max_threads);
}
num_threads = max_threads;
}
work  = (n + num_threads - 1)/num_threads;
ilow  = work * myid;
ihigh = ilow + work;
if (ihigh > n) ihigh = n;
#pragma omp parallel for  
for (iouter = 0; iouter < ihigh; iouter++) {
nzv = NONZER;
sprnvc(n, nzv, nn1, vc, ivc);
if (iouter >= ilow) {
vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
arow[iouter] = nzv;
for (ivelt = 0; ivelt < nzv; ivelt++) {
acol[iouter][ivelt] = ivc[ivelt] - 1;
aelt[iouter][ivelt] = vc[ivelt];
}
}
}
#pragma omp barrier
sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, 
aelt, firstrow, lastrow, last_n,
v, &iv[0], &iv[nz], RCOND, SHIFT);
}
static void sparse(double a[],
int colidx[],
int rowstr[],
int n,
int nz,
int nozer,
int arow[],
int acol[][NONZER+1],
double aelt[][NONZER+1],
int firstrow,
int lastrow,
int last_n[],
double v[],
int iv[],
int nzloc[],
double rcond,
double shift)
{
int nrows;
int i, j, j1, j2, nza, k, kk, nzrow, jcol;
double size, scale, ratio, va;
logical cont40;
nrows = lastrow - firstrow + 1;
j1 = ilow + 1;
j2 = ihigh + 1;
#pragma omp parallel for   
for (j = j1; j < j2; j++) {
rowstr[j] = 0;
}
for (i = 0; i < n; i++) {
for (nza = 0; nza < arow[i]; nza++) {
j = acol[i][nza];
if (j >= ilow && j < ihigh) {
j = j + 1;
rowstr[j] = rowstr[j] + arow[i];
}
}
}
if (myid == 0) {
rowstr[0] = 0;
j1 = 0;
}
for (j = j1+1; j < j2; j++) {
rowstr[j] = rowstr[j] + rowstr[j-1];
}
if (myid < num_threads) last_n[myid] = rowstr[j2-1];
#pragma omp barrier
nzrow = 0;
if (myid < num_threads) {
for (i = 0; i < myid; i++) {
nzrow = nzrow + last_n[i];
}
}
if (nzrow > 0) {
for (j = j1; j < j2; j++) {
rowstr[j] = rowstr[j] + nzrow;
}
}
#pragma omp barrier
nza = rowstr[nrows] - 1;
if (nza > nz) {
#pragma omp master
{
printf("Space for matrix elements exceeded in sparse\n");
printf("nza, nzmax = %d, %d\n", nza, nz);
}
exit(EXIT_FAILURE);
}
for (j = ilow; j < ihigh; j++) {
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
v[k] = 0.0;
iv[k] = -1;
}
nzloc[j] = 0;
}
size = 1.0;
ratio = pow(rcond, (1.0 / (double)(n)));
for (i = 0; i < n; i++) {
for (nza = 0; nza < arow[i]; nza++) {
j = acol[i][nza];
if (j < ilow || j >= ihigh) continue;
scale = size * aelt[i][nza];
for (nzrow = 0; nzrow < arow[i]; nzrow++) {
jcol = acol[i][nzrow];
va = aelt[i][nzrow] * scale;
if (jcol == j && j == i) {
va = va + rcond - shift;
}
cont40 = false;
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
if (iv[k] > jcol) {
for (kk = rowstr[j+1]-2; kk >= k; kk--) {
if (iv[kk] > -1) {
v[kk+1]  = v[kk];
iv[kk+1] = iv[kk];
}
}
iv[k] = jcol;
v[k]  = 0.0;
cont40 = true;
break;
} else if (iv[k] == -1) {
iv[k] = jcol;
cont40 = true;
break;
} else if (iv[k] == jcol) {
nzloc[j] = nzloc[j] + 1;
cont40 = true;
break;
}
}
if (cont40 == false) {
printf("internal error in sparse: i=%d\n", i);
exit(EXIT_FAILURE);
}
v[k] = v[k] + va;
}
}
size = size * ratio;
}
#pragma omp barrier
for (j = ilow+1; j < ihigh; j++) {
nzloc[j] = nzloc[j] + nzloc[j-1];
}
if (myid < num_threads) last_n[myid] = nzloc[ihigh-1];
#pragma omp barrier
nzrow = 0;
if (myid < num_threads) {
for (i = 0; i < myid; i++) {
nzrow = nzrow + last_n[i];
}
}
if (nzrow > 0) {
for (j = ilow; j < ihigh; j++) {
nzloc[j] = nzloc[j] + nzrow;
}
}
#pragma omp barrier
#pragma omp for
for (j = 0; j < nrows; j++) {
if (j > 0) {
j1 = rowstr[j] - nzloc[j-1];
} else {
j1 = 0;
}
j2 = rowstr[j+1] - nzloc[j];
nza = rowstr[j];
for (k = j1; k < j2; k++) {
a[k] = v[nza];
colidx[k] = iv[nza];
nza = nza + 1;
}
}
#pragma omp for
for (j = 1; j < nrows+1; j++) {
rowstr[j] = rowstr[j] - nzloc[j-1];
}
nza = rowstr[nrows] - 1;
}
static void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
int nzv, ii, i;
double vecelt, vecloc;
nzv = 0;
while (nzv < nz) {
vecelt = randlc(&tran, amult);
vecloc = randlc(&tran, amult);
i = icnvrt(vecloc, nn1) + 1;
if (i > n) continue;
logical was_gen = false;
for (ii = 0; ii < nzv; ii++) {
if (iv[ii] == i) {
was_gen = true;
break;
}
}
if (was_gen) continue;
v[nzv] = vecelt;
iv[nzv] = i;
nzv = nzv + 1;
}
}
static int icnvrt(double x, int ipwr2)
{
return (int)(ipwr2 * x);
}
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
int k;
logical set;
set = false;
for (k = 0; k < *nzv; k++) {
if (iv[k] == i) {
v[k] = val;
set  = true;
}
}
if (set == false) {
v[*nzv]  = val;
iv[*nzv] = i;
*nzv     = *nzv + 1;
}
}
