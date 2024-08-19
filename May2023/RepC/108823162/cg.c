#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "globals.h"
#include "randdp.h"
#include "timers.h"
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];
static double aelt[NAZ];
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
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
logical verified;
double zeta_verify_value, epsilon, err;
char *t_names[T_last];
for (i = 0; i < T_last; i++) {
timer_clear(i);
}
timer_start(T_init);
firstrow = 0;
lastrow  = NA-1;
firstcol = 0;
lastcol  = NA-1;
zeta_verify_value = VALID_RESULT;
printf("\nCG start...\n\n");
printf(" Size: %11d\n", NA);
printf(" Iterations: %5d\n", NITER);
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
iv);
#pragma omp parallel sections private(j) 
{
#pragma omp section
{
for (j = 0; j < lastrow - firstrow + 1; j++) {
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
colidx[k] = colidx[k] - firstcol;
}
}
}
#pragma omp section
{
#pragma omp parallel for
for (i = 0; i < NA+1; i++) {
x[i] = 1.0;
}
}
#pragma omp section
{
#pragma omp parallel for
for (j = 0; j < lastcol - firstcol + 1; j++) {
q[j] = 0.0;
z[j] = 0.0;
r[j] = 0.0;
p[j] = 0.0;
}
}
}
zeta = 0.0;
for (it = 1; it <= 1; it++) {
conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
norm_temp1 = 0.0;
norm_temp2 = 0.0;
for (j = 0; j < lastcol - firstcol + 1; j++) {
norm_temp1 = norm_temp1 + x[j] * z[j];
norm_temp2 = norm_temp2 + z[j] * z[j];
}
norm_temp2 = 1.0 / sqrt(norm_temp2);
#pragma omp parallel for
for (j = 0; j < lastcol - firstcol + 1; j++) {     
x[j] = norm_temp2 * z[j];
}
} 
#pragma omp parallel for
for (i = 0; i < NA+1; i++) {
x[i] = 1.0;
}
zeta = 0.0;
timer_stop(T_init);
printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));
timer_start(T_bench);
for (it = 1; it <= NITER; it++) {
if (timeron) timer_start(T_conj_grad);
conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
if (timeron) timer_stop(T_conj_grad);
norm_temp1 = 0.0;
norm_temp2 = 0.0;
for (j = 0; j < lastcol - firstcol + 1; j++) {
norm_temp1 = norm_temp1 + x[j]*z[j];
norm_temp2 = norm_temp2 + z[j]*z[j];
}
norm_temp2 = 1.0 / sqrt(norm_temp2);
zeta = SHIFT + 1.0 / norm_temp1;
if (it == 1) 
printf("\n   iteration           ||r||                 zeta\n");
printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);
#pragma omp parallel for
for (j = 0; j < lastcol - firstcol + 1; j++) {
x[j] = norm_temp2 * z[j];
}
} 
timer_stop(T_bench);
t = timer_read(T_bench);
printf("\nComplete...\n");
epsilon = 1.0e-10;
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
printf("\n\nExecution time : %lf seconds\n\n", t);
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
double d, sum, rho, rho0, alpha, beta;
rho = 0.0;
#pragma omp parallel for
for (j = 0; j < naa+1; j++) {
q[j] = 0.0;
z[j] = 0.0;
r[j] = x[j];
p[j] = r[j];
}
for (j = 0; j < lastcol - firstcol + 1; j++) 
rho = rho + r[j]*r[j];
for (cgit = 1; cgit <= cgitmax; cgit++) {
#pragma omp parallel for private(sum,k)
for (j = 0; j < lastrow - firstrow + 1; j++) {
sum = 0.0;
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
sum = sum + a[k]*p[colidx[k]];
}
q[j] = sum;
}
d = 0.0;
for (j = 0; j < lastcol - firstcol + 1; j++) {
d = d + p[j]*q[j];
}
alpha = rho / d;
rho0 = rho;
rho = 0.0;
#pragma omp parallel for
for (j = 0; j < lastcol - firstcol + 1; j++) {
z[j] = z[j] + alpha*p[j];  
r[j] = r[j] - alpha*q[j];
}
for (j = 0; j < lastcol - firstcol + 1; j++) {
rho = rho + r[j]*r[j];
}
beta = rho / rho0;
#pragma omp parallel for
for (j = 0; j < lastcol - firstcol + 1; j++) {
p[j] = r[j] + beta*p[j];
}
} 
sum = 0.0;
#pragma omp parallel for private(d)
for (j = 0; j < lastrow - firstrow + 1; j++) {
d = 0.0;
#pragma omp parallel for reduction(+:d)
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
d = d + a[k]*z[colidx[k]];
}
r[j] = d;
}
#pragma omp paralle for reduction(+:sum) private(d)
for (j = 0; j < lastcol-firstcol+1; j++) {
d   = x[j] - r[j];
sum = sum + d*d;
}
*rnorm = sqrt(sum);
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
int iv[])
{
int iouter, ivelt, nzv, nn1;
int ivc[NONZER+1];
double vc[NONZER+1];
nn1 = 1;
do {
nn1 = 2 * nn1;
} while (nn1 < n);
for (iouter = 0; iouter < n; iouter++) {
nzv = NONZER;
sprnvc(n, nzv, nn1, vc, ivc);
vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
arow[iouter] = nzv;
for (ivelt = 0; ivelt < nzv; ivelt++) {
acol[iouter][ivelt] = ivc[ivelt] - 1;
aelt[iouter][ivelt] = vc[ivelt];
}
}
sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, 
aelt, firstrow, lastrow,
iv, RCOND, SHIFT);
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
int nzloc[],
double rcond,
double shift)
{
int nrows;
int i, j, j1, j2, nza, k, kk, nzrow, jcol;
double size, scale, ratio, va;
logical cont40;
nrows = lastrow - firstrow + 1;
#pragma omp parallel for
for (j = 0; j < nrows+1; j++) {
rowstr[j] = 0;
}
for (i = 0; i < n; i++) {
for (nza = 0; nza < arow[i]; nza++) {
j = acol[i][nza] + 1;
rowstr[j] = rowstr[j] + arow[i];
}
}
rowstr[0] = 0;
for (j = 1; j < nrows+1; j++) {
rowstr[j] = rowstr[j] + rowstr[j-1];
}
nza = rowstr[nrows] - 1;
if (nza > nz) {
printf("Space for matrix elements exceeded in sparse\n");
printf("nza, nzmax = %d, %d\n", nza, nz);
exit(EXIT_FAILURE);
}
for (j = 0; j < nrows; j++) {
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
a[k] = 0.0;
colidx[k] = -1;
}
nzloc[j] = 0;
}
size = 1.0;
ratio = pow(rcond, (1.0 / (double)(n)));
for (i = 0; i < n; i++) {
for (nza = 0; nza < arow[i]; nza++) {
j = acol[i][nza];
scale = size * aelt[i][nza];
for (nzrow = 0; nzrow < arow[i]; nzrow++) {
jcol = acol[i][nzrow];
va = aelt[i][nzrow] * scale;
if (jcol == j && j == i) {
va = va + rcond - shift;
}
cont40 = false;
for (k = rowstr[j]; k < rowstr[j+1]; k++) {
if (colidx[k] > jcol) {
for (kk = rowstr[j+1]-2; kk >= k; kk--) {
if (colidx[kk] > -1) {
a[kk+1]  = a[kk];
colidx[kk+1] = colidx[kk];
}
}
colidx[k] = jcol;
a[k]  = 0.0;
cont40 = true;
break;
} else if (colidx[k] == -1) {
colidx[k] = jcol;
cont40 = true;
break;
} else if (colidx[k] == jcol) {
nzloc[j] = nzloc[j] + 1;
cont40 = true;
break;
}
}
if (cont40 == false) {
printf("internal error in sparse: i=%d\n", i);
exit(EXIT_FAILURE);
}
a[k] = a[k] + va;
}
}
size = size * ratio;
}
for (j = 1; j < nrows; j++) {
nzloc[j] = nzloc[j] + nzloc[j-1];
}
for (j = 0; j < nrows; j++) {
if (j > 0) {
j1 = rowstr[j] - nzloc[j-1];
} else {
j1 = 0;
}
j2 = rowstr[j+1] - nzloc[j];
nza = rowstr[j];
for (k = j1; k < j2; k++) {
a[k] = a[nza];
colidx[k] = colidx[nza];
nza = nza + 1;
}
}
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
