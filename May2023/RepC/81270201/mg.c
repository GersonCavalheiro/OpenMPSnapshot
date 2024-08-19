#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <omp.h>
typedef struct{
int N;
int Lmax;
int size[20];
double a[20];
double m;
double scale[20];
} param_t;
void relax(double *phi, double *res, int lev, int niter, param_t p);
void proj_res(double *res_c, double *rec_f, double *phi_f, int lev,param_t p);
void inter_add(double *phi_f, double *phi_c, int lev,param_t p);
double GetResRoot(double *phi, double *res, int lev, param_t p);
void relax_parallel(double *phi, double *res, int lev, int niter, param_t p);
void proj_res_parallel(double *res_c, double *res_f, double *phi_f,int lev,param_t p);
void inter_add_parallel(double *phi_f,double *phi_c,int lev,param_t p);
double GetResRoot_parallel(double *phi, double *res, int lev, param_t p);
struct timespec diff(struct timespec start, struct timespec end);
int main(int argc, char **argv)
{
struct timespec time1, time2, total;
double *phi[20], *res[20];
param_t p;
int nlev;
int i,lev;
p.Lmax = 6; 
p.N = 2*(int)pow(2,p.Lmax);  
p.m = 0.01;
nlev = 0; 
if(nlev  > p.Lmax){
printf("ERROR More levels than available in lattice! \n");
return 0;
}
clock_gettime(CLOCK_MONOTONIC, &time1);
printf("\n V cycle for %d by %d lattice with nlev = %d out of max  %d \n", p.N, p.N, nlev, p.Lmax);
p.size[0] = p.N;
p.a[0] = 1.0;
p.scale[0] = 1.0/(4.0 + p.m*p.m);
for(lev = 1;lev< p.Lmax+1; lev++) {
p.size[lev] = p.size[lev-1]/2;
p.a[lev] = 2.0 * p.a[lev-1];
p.scale[lev] = 1.0/(4.0 + p.m*p.m);
}
for(lev = 0;lev< p.Lmax+1; lev++) {
phi[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
res[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
for(i = 0;i< p.size[lev]*p.size[lev];i++) {
phi[lev][i] = 0.0;
res[lev][i] = 0.0;
}
}
res[0][p.N/2 + (p.N/2)*p.N] = 1.0*p.scale[0];  
double resmag = 1.0; 
int ncycle = 0;
int n_per_lev = 10;
resmag = GetResRoot_parallel(phi[0],res[0],0,p);
printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);
while(resmag > 0.00001) {
ncycle +=1;
for(lev = 0;lev<nlev; lev++) {  
relax_parallel(phi[lev],res[lev],lev, n_per_lev,p); 
proj_res_parallel(res[lev + 1], res[lev], phi[lev], lev,p);    
}
for(lev = nlev;lev >= 0; lev--) { 
relax_parallel(phi[lev],res[lev],lev, n_per_lev,p);   
if(lev > 0) inter_add_parallel(phi[lev-1], phi[lev], lev, p);   
}
resmag = GetResRoot_parallel(phi[0],res[0],0,p);
if (ncycle % 1000 == 0) printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);
}
clock_gettime(CLOCK_MONOTONIC, &time2);
total = diff(time1, time2);
printf("Total time: %ld.%ld seconds\n", total.tv_sec, total.tv_nsec);
clock_gettime(CLOCK_MONOTONIC, &time1);
printf("\n V cycle for %d by %d lattice with nlev = %d out of max  %d \n", p.N, p.N, nlev, p.Lmax);
p.size[0] = p.N;
p.a[0] = 1.0;
p.scale[0] = 1.0/(4.0 + p.m*p.m);
for(lev = 1;lev< p.Lmax+1; lev++) {
p.size[lev] = p.size[lev-1]/2;
p.a[lev] = 2.0 * p.a[lev-1];
p.scale[lev] = 1.0/(4.0 + p.m*p.m);
}
for(lev = 0;lev< p.Lmax+1; lev++) {
phi[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
res[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
for(i = 0;i< p.size[lev]*p.size[lev];i++) {
phi[lev][i] = 0.0;
res[lev][i] = 0.0;
}
}
res[0][p.N/2 + (p.N/2)*p.N] = 1.0*p.scale[0];  
resmag = 1.0; 
ncycle = 0;
n_per_lev = 10;
resmag = GetResRoot(phi[0],res[0],0,p);
printf("At the %d cycle the mag residue is %g \n",ncycle,resmag);
while(resmag > 0.00001) {
ncycle +=1;
for(lev = 0;lev<nlev; lev++) {  
relax(phi[lev],res[lev],lev, n_per_lev,p); 
proj_res(res[lev + 1], res[lev], phi[lev], lev,p);    
}
for(lev = nlev;lev >= 0; lev--) { 
relax(phi[lev],res[lev],lev, n_per_lev,p);   
if(lev > 0) inter_add(phi[lev-1], phi[lev], lev, p);   
}
resmag = GetResRoot(phi[0],res[0],0,p);
if (ncycle % 1000 == 0) printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);
}
clock_gettime(CLOCK_MONOTONIC, &time2);
total = diff(time1, time2);
printf("Total time: %ld.%ld seconds\n", total.tv_sec, total.tv_nsec);
return 0;
}
void relax_parallel(double *phi, double *res, int lev, int niter, param_t p) {
int i, x, y;
int L = p.size[lev];
double *tmp = malloc(L*L * sizeof(double));
#pragma omp parallel private(i,x,y) shared(p)
for(i=0; i < niter; i++) {
#pragma omp for
for(x = 0; x < L; x++) {
for(y = 0; y < L; y++) {
tmp[x + y*L] = res[x + y*L]
+ p.scale[lev] * (phi[(x+1)%L + y*L] + phi[(x-1+L)%L + y*L]
+ phi[x + ((y+1)%L)*L]  + phi[x + ((y-1+L)%L)*L]);
}
}
#pragma omp for
for(y = 0; y < L; y++) {
for(x = 0; x < L; x++) {
phi[x + y*L] = tmp[x + y*L];
}
}
}
return;
}
void relax(double *phi, double *res, int lev, int niter, param_t p) {
int i, x, y;
int L = p.size[lev];
double *tmp = malloc(L*L * sizeof(double));
for(i=0; i < niter; i++) {
for(x = 0; x < L; x++) {
for(y = 0; y < L; y++) {
tmp[x + y*L] = res[x + y*L]
+ p.scale[lev] * (phi[(x+1)%L + y*L] + phi[(x-1+L)%L + y*L]
+ phi[x + ((y+1)%L)*L]  + phi[x + ((y-1+L)%L)*L]);
}
}
for(y = 0; y < L; y++) {
for(x = 0; x < L; x++) {
phi[x + y*L] = tmp[x + y*L];
}
}
}
return;
}
void proj_res(double *res_c, double *res_f, double *phi_f,int lev,param_t p)
{
int L, Lc, x, y;
L = p.size[lev];
double r[L*L]; 
Lc = p.size[lev+1];  
for(x = 0; x< L; x++)
for(y = 0; y< L; y++)
r[x + y*L] = res_f[x + y*L] -  phi_f[x + y*L]
+ p.scale[lev]*(phi_f[(x+1)%L + y*L]
+ phi_f[(x-1+L)%L + y*L]
+ phi_f[x + ((y+1)%L)*L]
+ phi_f[x + ((y-1+L)%L)*L]);
for(x = 0; x< Lc; x++)
for(y = 0; y< Lc; y++)
res_c[x + y*Lc] = 0.25*(r[2*x + 2*y*L]
+ r[(2*x + 1)%L + 2*y*L]
+ r[2*x + ((2*y+1))%L*L]
+ r[(2*x+1)%L + ((2*y+1)%L)*L]);
return;
}
void proj_res_parallel(double *res_c, double *res_f, double *phi_f,int lev,param_t p)
{
int L, Lc, x, y;
L = p.size[lev];
double r[L*L]; 
Lc = p.size[lev+1];  
#pragma omp parallel for private(x, y) shared(p, L)
for(x = 0; x< L; x++)
for(y = 0; y< L; y++)
r[x + y*L] = res_f[x + y*L] -  phi_f[x + y*L]
+ p.scale[lev]*(phi_f[(x+1)%L + y*L]
+ phi_f[(x-1+L)%L + y*L]
+ phi_f[x + ((y+1)%L)*L]
+ phi_f[x + ((y-1+L)%L)*L]);
#pragma omp parallel for private(x, y) shared(p, L, Lc)
for(x = 0; x< Lc; x++)
for(y = 0; y< Lc; y++)
res_c[x + y*Lc] = 0.25*(r[2*x + 2*y*L]
+ r[(2*x + 1)%L + 2*y*L]
+ r[2*x + ((2*y+1))%L*L]
+ r[(2*x+1)%L + ((2*y+1)%L)*L]);
return;
}
void inter_add(double *phi_f,double *phi_c,int lev,param_t p)
{
int L, Lc, x, y;
Lc = p.size[lev];  
L = p.size[lev-1];
for(x = 0; x< Lc; x++)
for(y = 0; y < Lc; y++) {
phi_f[2*x  + 2*y*L]              += phi_c[x + y*Lc];
phi_f[(2*x + 1)%L   + 2*y*L]     += phi_c[x + y*Lc];
phi_f[2*x   + ((2*y+1))%L*L]     += phi_c[x + y*Lc];
phi_f[(2*x+1)%L + ((2*y+1)%L)*L] += phi_c[x + y*Lc];
}
for(x = 0; x< Lc; x++)
for(y = 0; y<Lc; y++)
phi_c[x + y*L] = 0.0;
return;
}
void inter_add_parallel(double *phi_f,double *phi_c,int lev,param_t p)
{
int L, Lc, x, y;
Lc = p.size[lev];  
L = p.size[lev-1];
#pragma omp parallel for private(x, y) shared(p, L, Lc)
for(x = 0; x< Lc; x++)
for(y = 0; y < Lc; y++) {
phi_f[2*x  + 2*y*L]              += phi_c[x + y*Lc];
phi_f[(2*x + 1)%L   + 2*y*L]     += phi_c[x + y*Lc];
phi_f[2*x   + ((2*y+1))%L*L]     += phi_c[x + y*Lc];
phi_f[(2*x+1)%L + ((2*y+1)%L)*L] += phi_c[x + y*Lc];
}
#pragma omp parallel for private(x, y) shared(p, L, Lc)
for(x = 0; x< Lc; x++)
for(y = 0; y<Lc; y++)
phi_c[x + y*L] = 0.0;
return;
}
double GetResRoot(double *phi, double *res, int lev, param_t p)
{
int x, y;
double residue;
double ResRoot = 0.0;
int L;
L  = p.size[lev];
for(x = 0; x < L; x++)
for(y = 0; y<L; y++) {
residue = res[x + y*L]/p.scale[lev] - phi[x + y*L]/p.scale[lev]
+ (phi[(x+1)%L + y*L] + phi[(x-1+L)%L + y*L]
+  phi[x + ((y+1)%L)*L]  + phi[x + ((y-1+L)%L)*L]);
ResRoot += residue*residue; 
}
return sqrt(ResRoot);
}
double GetResRoot_parallel(double *phi, double *res, int lev, param_t p)
{
int x, y;
double residue;
double ResRoot = 0.0;
int L;
L  = p.size[lev];
#pragma omp parallel for private(y,x,residue) shared(p, L) reduction(+:ResRoot)
for(x = 0; x < L; x++)
for(y = 0; y<L; y++) {
residue = res[x + y*L]/p.scale[lev] - phi[x + y*L]/p.scale[lev]
+ (phi[(x+1)%L + y*L] + phi[(x-1+L)%L + y*L]
+  phi[x + ((y+1)%L)*L]  + phi[x + ((y-1+L)%L)*L]);
ResRoot += residue*residue; 
}
return sqrt(ResRoot);
}
struct timespec diff(struct timespec start, struct timespec end)
{
struct timespec temp;
if ((end.tv_nsec-start.tv_nsec)<0) {
temp.tv_sec = end.tv_sec-start.tv_sec-1;
temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
} else {
temp.tv_sec = end.tv_sec-start.tv_sec;
temp.tv_nsec = end.tv_nsec-start.tv_nsec;
}
return temp;
}
