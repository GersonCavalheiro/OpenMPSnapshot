#include "../common/npb-C.h"
#include "npbparams.h"
#define	NZ	NA*(NONZER+1)*(NONZER+1)+NA*(NONZER+2)
#include <omp.h> 
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static int colidx[2198001];
static int rowstr[14002];
static int iv[28002];
static int arow[2198001];
static int acol[2198001];
static double v[14002];
static double aelt[2198001];
static double a[2198001];
static double x[14003];
static double z[14003];
static double p[14003];
static double q[14003];
static double r[14003];
static double amult;
static double tran;
static 
void conj_grad(int colidx[],int rowstr[],double x[],double z[],double a[],double p[],double q[],double r[],double *rnorm);
static void makea(int n,int nz,double a[],int colidx[],int rowstr[],int nonzer,int firstrow,int lastrow,int firstcol,int lastcol,double rcond,int arow[],int acol[],double aelt[],double v[],int iv[],double shift);
static void sparse(double a[],int colidx[],int rowstr[],int n,int arow[],int acol[],double aelt[],int firstrow,int lastrow,double x[],boolean mark[],int nzloc[],int nnza);
static void sprnvc(int n,int nz,double v[],int iv[],int nzloc[],int mark[]);
static int icnvrt(double x,int ipwr2);
static void vecset(int n,double v[],int iv[],int *nzv,int i,double val);
int main(int argc,char **argv)
{
int i;
int j;
int k;
int it;
int nthreads = 1;
double zeta;
double rnorm;
double norm_temp11;
double norm_temp12;
double t;
double mflops;
char class;
boolean verified;
double zeta_verify_value;
double epsilon;
firstrow = 1;
lastrow = 14000;
firstcol = 1;
lastcol = 14000;
if (14000 == 1400 && 11 == 7 && 15 == 15 && 20.0 == 10.0) {
class = 'S';
zeta_verify_value = 8.5971775078648;
}
else if (14000 == 7000 && 11 == 8 && 15 == 15 && 20.0 == 12.0) {
class = 'W';
zeta_verify_value = 10.362595087124;
}
else if (14000 == 14000 && 11 == 11 && 15 == 15 && 20.0 == 20.0) {
class = 'A';
zeta_verify_value = 17.130235054029;
}
else if (14000 == 75000 && 11 == 13 && 15 == 75 && 20.0 == 60.0) {
class = 'B';
zeta_verify_value = 22.712745482631;
}
else if (14000 == 150000 && 11 == 15 && 15 == 75 && 20.0 == 110.0) {
class = 'C';
zeta_verify_value = 28.973605592845;
}
else {
class = 'U';
}
printf("\n\n NAS Parallel Benchmarks 3.0 structured OpenMP C version - CG Benchmark\n");
printf(" Size: %10d\n",14000);
printf(" Iterations: %5d\n",15);
naa = 14000;
nzz = 14000 * (11 + 1) * (11 + 1) + 14000 * (11 + 2);
tran = 314159265.0;
amult = 1220703125.0;
zeta = randlc(&tran,amult);
makea(naa,nzz,a,colidx,rowstr,11,firstrow,lastrow,firstcol,lastcol,1.0e-1,arow,acol,aelt,v,iv,20.0);
{
for (j = 1; j <= lastrow - firstrow + 1; j += 1) {
#pragma omp parallel for private (k)
for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k += 1) {
colidx[k] = colidx[k] - firstcol + 1;
}
}
#pragma omp parallel for private (i)
for (i = 1; i <= 14001; i += 1) {
x[i] = 1.0;
}
#pragma omp parallel for private (j)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
q[j] = 0.0;
z[j] = 0.0;
r[j] = 0.0;
p[j] = 0.0;
}
}
zeta = 0.0;
for (it = 1; it <= 1; it += 1) {
conj_grad(colidx,rowstr,x,z,a,p,q,r,&rnorm);
norm_temp11 = 0.0;
norm_temp12 = 0.0;
#pragma omp parallel for private (j) reduction (+:norm_temp11,norm_temp12)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
norm_temp11 = norm_temp11 + x[j] * z[j];
norm_temp12 = norm_temp12 + z[j] * z[j];
}
norm_temp12 = 1.0 / sqrt(norm_temp12);
#pragma omp parallel for private (j) firstprivate (norm_temp12)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
x[j] = norm_temp12 * z[j];
}
}
#pragma omp parallel for private (i)
for (i = 1; i <= 14001; i += 1) {
x[i] = 1.0;
}
zeta = 0.0;
timer_clear(1);
timer_start(1);
for (it = 1; it <= 15; it += 1) {
conj_grad(colidx,rowstr,x,z,a,p,q,r,&rnorm);
norm_temp11 = 0.0;
norm_temp12 = 0.0;
#pragma omp parallel for private (j) reduction (+:norm_temp11,norm_temp12)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
norm_temp11 = norm_temp11 + x[j] * z[j];
norm_temp12 = norm_temp12 + z[j] * z[j];
}
norm_temp12 = 1.0 / sqrt(norm_temp12);
zeta = 20.0 + 1.0 / norm_temp11;
if (it == 1) {
printf("   iteration           ||r||                 zeta\n");
}
printf("    %5d       %20.14e%20.13e\n",it,rnorm,zeta);
#pragma omp parallel for private (j) firstprivate (norm_temp12)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
x[j] = norm_temp12 * z[j];
}
}
{
}
timer_stop(1);
t = timer_read(1);
printf(" Benchmark completed\n");
epsilon = 1.0e-10;
if (class != 'U') {
if (fabs(zeta - zeta_verify_value) <= epsilon) {
verified = 1;
printf(" VERIFICATION SUCCESSFUL\n");
printf(" Zeta is    %20.12e\n",zeta);
printf(" Error is   %20.12e\n",zeta - zeta_verify_value);
}
else {
verified = 0;
printf(" VERIFICATION FAILED\n");
printf(" Zeta                %20.12e\n",zeta);
printf(" The correct zeta is %20.12e\n",zeta_verify_value);
}
}
else {
verified = 0;
printf(" Problem size unknown\n");
printf(" NO VERIFICATION PERFORMED\n");
}
if (t != 0.0) {
mflops = 2.0 * 15 * 14000 * (3.0 + (11 * (11 + 1)) + 25.0 * (5.0 + (11 * (11 + 1))) + 3.0) / t / 1000000.0;
}
else {
mflops = 0.0;
}
c_print_results("CG",class,14000,0,0,15,nthreads,t,mflops,"          floating point",verified,"3.0 structured","14 Jan 2020","(none)","(none)","-lm","(none)","(none)","(none)","randdp");
}
static void conj_grad(
int colidx[],
int rowstr[],
double x[],
double z[],
double a[],
double p[],
double q[],
double r[],
double *rnorm)
{
static int callcount = 0;
double d;
double sum;
double rho;
double rho0;
double alpha;
double beta;
int i;
int j;
int k;
int cgit;
int cgitmax = 25;
rho = 0.0;
{
#pragma omp parallel for private (j) firstprivate (naa)
for (j = 1; j <= naa + 1; j += 1) {
q[j] = 0.0;
z[j] = 0.0;
r[j] = x[j];
p[j] = r[j];
}
#pragma omp parallel for private (j) reduction (+:rho)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
rho = rho + r[j] * r[j];
}
}
for (cgit = 1; cgit <= cgitmax; cgit += 1) {
rho0 = rho;
d = 0.0;
rho = 0.0;
{
#pragma omp parallel for private (sum,j,k)
for (j = 1; j <= lastrow - firstrow + 1; j += 1) {
sum = 0.0;
#pragma omp parallel for private (k) reduction (+:sum)
for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k += 1) {
sum = sum + a[k] * p[colidx[k]];
}
q[j] = sum;
}
#pragma omp parallel for private (j) reduction (+:d)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
d = d + p[j] * q[j];
}
alpha = rho0 / d;
#pragma omp parallel for private (j) reduction (+:rho) firstprivate (alpha)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
z[j] = z[j] + alpha * p[j];
r[j] = r[j] - alpha * q[j];
rho = rho + r[j] * r[j];
}
beta = rho / rho0;
#pragma omp parallel for private (j) firstprivate (beta)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
p[j] = r[j] + beta * p[j];
}
callcount++;
}
}
sum = 0.0;
{
#pragma omp parallel for private (d,j,k) firstprivate (firstrow,lastrow)
for (j = 1; j <= lastrow - firstrow + 1; j += 1) {
d = 0.0;
#pragma omp parallel for private (k) reduction (+:d)
for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k += 1) {
d = d + a[k] * z[colidx[k]];
}
r[j] = d;
}
#pragma omp parallel for private (d,j) reduction (+:sum) firstprivate (firstcol,lastcol)
for (j = 1; j <= lastcol - firstcol + 1; j += 1) {
d = x[j] - r[j];
sum = sum + d * d;
}
}
*rnorm = sqrt(sum);
}
static void makea(int n,int nz,
double a[],
int colidx[],
int rowstr[],int nonzer,int firstrow,int lastrow,int firstcol,int lastcol,double rcond,
int arow[],
int acol[],
double aelt[],
double v[],
int iv[],double shift)
{
int i;
int nnza;
int iouter;
int ivelt;
int ivelt1;
int irow;
int nzv;
double size;
double ratio;
double scale;
int jcol;
size = 1.0;
ratio = pow(rcond,1.0 / ((double )n));
nnza = 0;
#pragma omp parallel for private (i)
for (i = 1; i <= n; i += 1) {
colidx[n + i] = 0;
}
for (iouter = 1; iouter <= n; iouter += 1) {
nzv = nonzer;
sprnvc(n,nzv,v,iv,&colidx[0],&colidx[n]);
vecset(n,v,iv,&nzv,iouter,0.5);
for (ivelt = 1; ivelt <= nzv; ivelt += 1) {
jcol = iv[ivelt];
if (jcol >= firstcol && jcol <= lastcol) {
scale = size * v[ivelt];
for (ivelt1 = 1; ivelt1 <= nzv; ivelt1 += 1) {
irow = iv[ivelt1];
if (irow >= firstrow && irow <= lastrow) {
nnza = nnza + 1;
if (nnza > nz) {
printf("Space for matrix elements exceeded in makea\n");
printf("nnza, nzmax = %d, %d\n",nnza,nz);
printf("iouter = %d\n",iouter);
exit(1);
}
acol[nnza] = jcol;
arow[nnza] = irow;
aelt[nnza] = v[ivelt1] * scale;
}
}
}
}
size = size * ratio;
}
for (i = firstrow; i <= lastrow; i += 1) {
if (i >= firstcol && i <= lastcol) {
iouter = n + i;
nnza = nnza + 1;
if (nnza > nz) {
printf("Space for matrix elements exceeded in makea\n");
printf("nnza, nzmax = %d, %d\n",nnza,nz);
printf("iouter = %d\n",iouter);
exit(1);
}
acol[nnza] = i;
arow[nnza] = i;
aelt[nnza] = rcond - shift;
}
}
sparse(a,colidx,rowstr,n,arow,acol,aelt,firstrow,lastrow,v,&iv[0],&iv[n],nnza);
}
static void sparse(
double a[],
int colidx[],
int rowstr[],int n,
int arow[],
int acol[],
double aelt[],int firstrow,int lastrow,
double x[],
boolean mark[],
int nzloc[],int nnza)
{
int nrows;
int i;
int j;
int jajp1;
int nza;
int k;
int nzrow;
double xi;
nrows = lastrow - firstrow + 1;
#pragma omp parallel for private (j)
for (j = 1; j <= n; j += 1) {
rowstr[j] = 0;
mark[j] = 0;
}
rowstr[n + 1] = 0;
for (nza = 1; nza <= nnza; nza += 1) {
j = arow[nza] - firstrow + 1 + 1;
rowstr[j] = rowstr[j] + 1;
}
rowstr[1] = 1;
for (j = 2; j <= nrows + 1; j += 1) {
rowstr[j] = rowstr[j] + rowstr[j - 1];
}
for (j = 0; j <= nrows - 1; j += 1) {
#pragma omp parallel for private (k)
for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k += 1) {
a[k] = 0.0;
}
}
for (nza = 1; nza <= nnza; nza += 1) {
j = arow[nza] - firstrow + 1;
k = rowstr[j];
a[k] = aelt[nza];
colidx[k] = acol[nza];
rowstr[j] = rowstr[j] + 1;
}
for (j = nrows; j >= 1; j += -1) {
rowstr[j + 1] = rowstr[j];
}
rowstr[1] = 1;
nza = 0;
#pragma omp parallel for private (i) firstprivate (n)
for (i = 1; i <= n; i += 1) {
x[i] = 0.0;
mark[i] = 0;
}
jajp1 = rowstr[1];
for (j = 1; j <= nrows; j += 1) {
nzrow = 0;
for (k = jajp1; k <= rowstr[j + 1] - 1; k += 1) {
i = colidx[k];
x[i] = x[i] + a[k];
if (mark[i] == 0 && x[i] != 0.0) {
mark[i] = 1;
nzrow = nzrow + 1;
nzloc[nzrow] = i;
}
}
for (k = 1; k <= nzrow; k += 1) {
i = nzloc[k];
mark[i] = 0;
xi = x[i];
x[i] = 0.0;
if (xi != 0.0) {
nza = nza + 1;
a[nza] = xi;
colidx[nza] = i;
}
}
jajp1 = rowstr[j + 1];
rowstr[j + 1] = nza + rowstr[1];
}
}
static void sprnvc(int n,int nz,
double v[],
int iv[],
int nzloc[],
int mark[])
{
int nn1;
int nzrow;
int nzv;
int ii;
int i;
double vecelt;
double vecloc;
nzv = 0;
nzrow = 0;
nn1 = 1;
do {
nn1 = 2 * nn1;
}while (nn1 < n);
while(nzv < nz){
vecelt = randlc(&tran,amult);
vecloc = randlc(&tran,amult);
i = icnvrt(vecloc,nn1) + 1;
if (i > n) 
continue; 
if (mark[i] == 0) {
mark[i] = 1;
nzrow = nzrow + 1;
nzloc[nzrow] = i;
nzv = nzv + 1;
v[nzv] = vecelt;
iv[nzv] = i;
}
}
for (ii = 1; ii <= nzrow; ii += 1) {
i = nzloc[ii];
mark[i] = 0;
}
}
static int icnvrt(double x,int ipwr2)
{
return (int )(ipwr2 * x);
}
static void vecset(int n,
double v[],
int iv[],int *nzv,int i,double val)
{
int k;
boolean set;
set = 0;
#pragma omp parallel for private (k)
for (k = 1; k <=  *nzv; k += 1) {
if (iv[k] == i) {
v[k] = val;
set = 1;
}
}
if (set == 0) {
*nzv =  *nzv + 1;
v[ *nzv] = val;
iv[ *nzv] = i;
}
}
