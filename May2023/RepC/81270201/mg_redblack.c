#include "mg_serial.h"
void relax(double *phi, double *res, int lev, int niter, param_t p);
void proj_res(double *res_c, double *rec_f, double *phi_f, int lev,param_t p);
void inter_add(double *phi_f, double *phi_c, int lev, param_t p);
double GetResRoot(double *phi, double *res, int lev, param_t p);
int serial_multigrid(int Lmax)
{
double *phi[20], *res[20];
param_t p;
int nlev;
int i,j,lev;
p.Lmax = Lmax; 
p.N = 2*(int)pow(2,p.Lmax);  
p.m = 0.01;
nlev = Lmax-1; 
if(nlev  > p.Lmax)
{
printf("  ERROR: More levels than available in lattice! \n");
return 1;
}
double resmag = 1.0; 
int ncycle = 0;
int n_per_lev = 10;
printf("  V cycle for %d by %d lattice with nlev = %d out of max %d \n", p.N, p.N, nlev, p.Lmax);
p.size[0] = p.N;
p.a[0] = 1.0;
p.scale[0] = 1.0/(4.0 + p.m*p.m);
for(lev = 1;lev< p.Lmax+1; lev++)
{
p.size[lev] = (p.size[lev-1])/2;
p.a[lev] = 2.0 * p.a[lev-1];
p.scale[lev] = 1.0/(4.0 + p.m*p.m*p.a[lev]*p.a[lev]);
}
for(lev = 0;lev< p.Lmax+1; lev++)
{
phi[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
res[lev] = (double *) malloc(p.size[lev]*p.size[lev] * sizeof(double));
for(i = 0; i < p.size[lev]*p.size[lev]; i++)
{
phi[lev][i] = 0.0;
res[lev][i] = 0.0;
}
}
res[0][(1*(p.N/4)) + ((1*(p.N/4)))*p.N] = 1.0*p.scale[0];
res[0][(1*(p.N/4)) + ((3*(p.N/4)))*p.N] = 1.0*p.scale[0];
res[0][(3*(p.N/4)) + ((1*(p.N/4)))*p.N] = 1.0*p.scale[0];
res[0][(3*(p.N/4)) + ((3*(p.N/4)))*p.N] = 1.0*p.scale[0];
resmag = 1.0; 
ncycle = 0;
n_per_lev = 10;
resmag = GetResRoot(phi[0],res[0],0,p);
printf("    At the %d cycle the mag residue is %g \n",ncycle,resmag);
while(resmag > 0.00001)
{
ncycle +=1;
for(lev = 0;lev<nlev; lev++)
{  
relax(phi[lev],res[lev],lev, n_per_lev,p); 
proj_res(res[lev + 1], res[lev], phi[lev], lev,p);    
}
for(lev = nlev;lev >= 0; lev--)
{ 
relax(phi[lev],res[lev],lev, n_per_lev,p);   
if(lev > 0) inter_add(phi[lev-1], phi[lev], lev, p);   
}
resmag = GetResRoot(phi[0],res[0],0,p);
if (ncycle % 500 == 0) printf("    At the %d cycle the mag residue is %g \n", ncycle, resmag);
}
printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);
FILE *file = fopen("serial_data.dat", "w+");
for (i=0; i< p.N; i++)
{
for (j=0; j<p.N; j++)
{
fprintf(file, "%i %i %f\n", i, j, phi[0][i + j*p.N]);
}
}
return 0;
}
void relax(double *phi, double *res, int lev, int niter, param_t p) {
int i, j;
int L = p.size[lev];
for(int iter=0; iter < niter; i++) {
for (i=1; i<L-1; i++) {
for (j=1; j<L-1; j+=2) {
int shift;
if (i%2 == 0) { shift = 1; }
else { shift = 0; }
int index       = getIndex(i, j+shift, m);
int pos_l  = getIndex(i, j-1+shift, m);
int pos_r = getIndex(i, j+1+shift, m);
int pos_u    = getIndex(i+1, j+shift, m);
int pos_d  = getIndex(i-1, j+shift, m);
phi[index] = p.scale[lev] * (phi[pos_l] + phi[pos_r]  +
phi[pos_u] + phi[pos_d]) +
res[index];
}
}
for (i=1; i<L-1; i++) {
for (j=2; j<L-1; j+=2) {
int shift;
if (i%2 == 0) { shift = 1; }
else { shift = 0; }
int index       = getIndex(i, j+shift, m);
int pos_l  = getIndex(i, j-1+shift, m);
int pos_r = getIndex(i, j+1+shift, m);
int pos_u    = getIndex(i+1, j+shift, m);
int pos_d  = getIndex(i-1, j+shift, m);
phi[index] = p.scale[lev] * (phi[pos_l] + phi[pos_r]  +
phi[pos_u] + phi[pos_d]) +
res[index];
}
}
}
return;
}
void proj_res(double *res_c, double *res_f, double *phi_f, int lev, param_t p)
{
int L, Lc, x, y;
L = p.size[lev];
double r[L*L]; 
Lc = p.size[lev+1];  
double left, right, up, down;
for(x = 0; x< L; x++) {
for(y = 0; y< L; y++) {
left  = (x == 0)   ? res_f[    y*L] : phi_f[(x-1) +  y   *L];
right = (x == L-1) ? res_f[x + y*L] : phi_f[(x+1) +  y   *L];
up    = (y == 0)   ? res_f[x      ] : phi_f[ x    + (y-1)*L];
down  = (y == L-1) ? res_f[x + y*L] : phi_f[ x    + (y+1)*L];
r[x + y*L] = res_f[x + y*L] -  phi_f[x + y*L] + p.scale[lev]*(left + right + up + down);
}
}
for(x = 0; x< Lc; x++) {
for(y = 0; y< Lc; y++) {
res_c[x + y*Lc] = 0.25*(r[ 2*x      +  2*y   *L] +
r[(2*x + 1) +  2*y   *L] +
r[ 2*x      + (2*y+1)*L] +
r[(2*x+1)%L + (2*y+1)*L]);
}
}
return;
}
void inter_add(double *phi_f,double *phi_c,int lev,param_t p)
{
int L, Lc, x, y;
Lc = p.size[lev];  
L = p.size[lev-1];
for(x = 0; x< Lc; x++) {
for(y = 0; y < Lc; y++) {
phi_f[ 2*x      +  2*y   *L] += phi_c[x + y*Lc];
phi_f[(2*x + 1) +  2*y   *L] += phi_c[x + y*Lc];
phi_f[ 2*x      + (2*y+1)*L] += phi_c[x + y*Lc];
phi_f[(2*x + 1) + (2*y+1)*L] += phi_c[x + y*Lc];
}
}
for(x = 0; x< Lc; x++) {
for(y = 0; y<Lc; y++) {
phi_c[x + y*L] = 0.0;
}
}
return;
}
double GetResRoot(double *phi, double *res, int lev, param_t p)
{
int x, y;
double residue;
double ResRoot = 0.0;
int L;
double left, right, up, down;
L  = p.size[lev];
for(x = 0; x < L; x++) {
for(y = 0; y<L; y++) {
left  = (x == 0)   ? res[    y*L] : phi[(x-1) +  y   *L];
right = (x == L-1) ? res[x + y*L] : phi[(x+1) +  y   *L];
up    = (y == 0)   ? res[x      ] : phi[ x    + (y-1)*L];
down  = (y == L-1) ? res[x + y*L] : phi[ x    + (y+1)*L];
residue = res[x + y*L]/p.scale[lev] - phi[x + y*L]/p.scale[lev] + (left + right + up + down);
ResRoot += residue*residue; 
}
}
return sqrt(ResRoot);
}
