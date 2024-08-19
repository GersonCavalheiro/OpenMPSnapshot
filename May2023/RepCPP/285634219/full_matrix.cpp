#include <math.h>
#include <stdio.h>

struct full_data
{
int sizex;
int sizey;
int Nmats;
double * __restrict rho;
double * __restrict rho_mat_ave;
double * __restrict p;
double * __restrict Vf;
double * __restrict t;
double * __restrict V;
double * __restrict x;
double * __restrict y;
double * __restrict n;
double * __restrict rho_ave;
};

void full_matrix_cell_centric(full_data cc)
{
int sizex = cc.sizex;
int sizey = cc.sizey;
int Nmats = cc.Nmats;
double * __restrict Vf = cc.Vf;
double * __restrict V = cc.V;
double * __restrict rho = cc.rho;
double * __restrict rho_ave = cc.rho_ave;
double * __restrict p = cc.p;
double * __restrict t = cc.t;
double * __restrict x = cc.x;
double * __restrict y = cc.y;
double * __restrict n = cc.n;
double * __restrict rho_mat_ave = cc.rho_mat_ave;

#if defined(NACC)
#pragma acc data copy(rho[0:sizex*sizey*Nmats], p[0:sizex*sizey*Nmats], t[0:sizex*sizey*Nmats], Vf[0:sizex*sizey*Nmats]) \
copy(V[0:sizex*sizey],x[0:sizex*sizey],y[0:sizex*sizey],n[0:Nmats],rho_ave[0:sizex*sizey]) \
copy(rho_mat_ave[0:sizex*sizey*Nmats])
#endif
{
#if defined(OMP)
#pragma omp parallel for collapse(2)
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int j = 0; j < sizey; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 0; i < sizex; i++){
double ave = 0.0;
for (int mat = 0; mat < Nmats; mat++) {
if (Vf[(i+sizex*j)*Nmats+mat] > 0.0)
ave += rho[(i+sizex*j)*Nmats+mat]*Vf[(i+sizex*j)*Nmats+mat];
}
rho_ave[i+sizex*j] = ave/V[i+sizex*j];
}
}
#ifdef DEBUG
printf("Full matrix, cell centric, alg 1: %g sec\n", omp_get_wtime()-t1);
#endif

#if defined(OMP)
#pragma omp parallel for collapse(2)
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int j = 0; j < sizey; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 0; i < sizex; i++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int mat = 0; mat < Nmats; mat++) {
if (Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
double nm = n[mat];
p[(i+sizex*j)*Nmats+mat] = (nm * rho[(i+sizex*j)*Nmats+mat] * t[(i+sizex*j)*Nmats+mat]) / Vf[(i+sizex*j)*Nmats+mat];
}
else {
p[(i+sizex*j)*Nmats+mat] = 0.0;
}
}
}
}
#ifdef DEBUG
printf("Full matrix, cell centric, alg 2: %g sec\n", omp_get_wtime()-t1);
#endif

#if defined(OMP)
#pragma omp parallel for collapse(2)
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int j = 1; j < sizey-1; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 1; i < sizex-1; i++) {
double xo = x[i+sizex*j];
double yo = y[i+sizex*j];

double dsqr[9];

for (int nj = -1; nj <= 1; nj++) {
for (int ni = -1; ni <= 1; ni++) {

dsqr[(nj+1)*3 + (ni+1)] = 0.0;

double xi = x[(i+ni)+sizex*(j+nj)];
double yi = y[(i+ni)+sizex*(j+nj)];

dsqr[(nj+1)*3 + (ni+1)] += (xo - xi) * (xo - xi);
dsqr[(nj+1)*3 + (ni+1)] += (yo - yi) * (yo - yi);
}
}
#pragma omp simd
for (int mat = 0; mat < Nmats; mat++) {
if (Vf[(i+sizex*j)*Nmats+mat] > 0.0) {
double rho_sum = 0.0;
int Nn = 0;

for (int nj = -1; nj <= 1; nj++) {
if ((j + nj < 0) || (j + nj >= sizey)) 
continue;

for (int ni = -1; ni <= 1; ni++) {
if ((i + ni < 0) || (i + ni >= sizex)) 
continue;

if (Vf[((i+ni)+sizex*(j+nj))*Nmats+mat] > 0.0) {
rho_sum += rho[((i+ni)+sizex*(j+nj))*Nmats+mat] / dsqr[(nj+1)*3 + (ni+1)];
Nn += 1;
}
}
}
rho_mat_ave[(i+sizex*j)*Nmats+mat] = rho_sum / Nn;
}
else {
rho_mat_ave[(i+sizex*j)*Nmats+mat] = 0.0;
}
}
}
}
#ifdef DEBUG
printf("Full matrix, cell centric, alg 3: %g sec\n", omp_get_wtime()-t1);
#endif
}
}

void full_matrix_material_centric(full_data cc, full_data mc)
{
int sizex = mc.sizex;
int sizey = mc.sizey;
int Nmats = mc.Nmats;
int ncells = sizex * sizey;
double * __restrict Vf = mc.Vf;
double * __restrict V = mc.V;
double * __restrict rho = mc.rho;
double * __restrict rho_ave = mc.rho_ave;
double * __restrict p = mc.p;
double * __restrict t = mc.t;
double * __restrict x = mc.x;
double * __restrict y = mc.y;
double * __restrict n = mc.n;
double * __restrict rho_mat_ave = mc.rho_mat_ave;
#if defined(NACC)
#pragma acc data copy(rho[0:sizex*sizey*Nmats], p[0:sizex*sizey*Nmats], t[0:sizex*sizey*Nmats], Vf[0:sizex*sizey*Nmats]) \
copy(V[0:sizex*sizey],x[0:sizex*sizey],y[0:sizex*sizey],n[0:Nmats],rho_ave[0:sizex*sizey]) \
copy(rho_mat_ave[0:sizex*sizey*Nmats])
#endif
{
#if defined(OMP)
#pragma omp parallel for 
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int j = 0; j < sizey; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 0; i < sizex; i++) {
rho_ave[i+sizex*j] = 0.0;
}
}

for (int mat = 0; mat < Nmats; mat++) {
#if defined(OMP)
#pragma omp parallel for 
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int j = 0; j < sizey; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 0; i < sizex; i++) {
if (Vf[ncells*mat + i+sizex*j] > 0.0)
rho_ave[i+sizex*j] += rho[ncells*mat + i+sizex*j] * Vf[ncells*mat + i+sizex*j];
}
}
}

#if defined(OMP)
#pragma omp parallel for 
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int j = 0; j < sizey; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 0; i < sizex; i++) {
rho_ave[i+sizex*j] /= V[i+sizex*j];
}
}
#ifdef DEBUG
printf("Full matrix, material centric, alg 1: %g sec\n", omp_get_wtime()-t1);
#endif

#if defined(OMP)
#pragma omp parallel for collapse(2)
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int mat = 0; mat < Nmats; mat++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int j = 0; j < sizey; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int i = 0; i < sizex; i++) {
double nm = n[mat];
if (Vf[ncells*mat + i+sizex*j] > 0.0) {
p[ncells*mat + i+sizex*j] = (nm * rho[ncells*mat + i+sizex*j] * t[ncells*mat + i+sizex*j]) / Vf[ncells*mat + i+sizex*j];
}
else {
p[ncells*mat + i+sizex*j] = 0.0;
}
}
}
}
#ifdef DEBUG
printf("Full matrix, material centric, alg 2: %g sec\n", omp_get_wtime()-t1);
#endif

#if defined(OMP)
#pragma omp parallel for collapse(2)
#elif defined(NACC)
#pragma acc parallel
#pragma acc loop independent
#endif
for (int mat = 0; mat < Nmats; mat++) {
#if defined(NACC)
#pragma acc loop independent
#endif
for (int j = 1; j < sizey-1; j++) {
#if defined(NACC)
#pragma acc loop independent
#endif
#pragma omp simd
for (int i = 1; i < sizex-1; i++) {
if (Vf[ncells*mat + i+sizex*j] > 0.0) {
double xo = x[i+sizex*j];
double yo = y[i+sizex*j];

double rho_sum = 0.0;
int Nn = 0;

for (int nj = -1; nj <= 1; nj++) {
if ((j + nj < 0) || (j + nj >= sizey)) 
continue;

for (int ni = -1; ni <= 1; ni++) {
if ((i + ni < 0) || (i + ni >= sizex)) 
continue;

if (Vf[ncells*mat + (i+ni)+sizex*(j+nj)] > 0.0) {
double dsqr = 0.0;

double xi = x[(i+ni)+sizex*(j+nj)];
double yi = y[(i+ni)+sizex*(j+nj)];

dsqr += (xo - xi) * (xo - xi);
dsqr += (yo - yi) * (yo - yi);

rho_sum += rho[ncells*mat + (i+ni)+sizex*(j+nj)] / dsqr;
Nn += 1;
}
}
}

rho_mat_ave[ncells*mat + i+sizex*j] = rho_sum / Nn;
}
else {
rho_mat_ave[ncells*mat + i+sizex*j] = 0.0;
}
}
}
}
#ifdef DEBUG
printf("Full matrix, material centric, alg 2: %g sec\n", omp_get_wtime()-t1);
#endif
}
}

bool full_matrix_check_results(full_data cc, full_data mc)
{

int sizex = cc.sizex;
int sizey = cc.sizey;
int Nmats = cc.Nmats;
int ncells = sizex * sizey;
#ifdef DEBUG
printf("Checking results of full matrix representation... ");
#endif

for (int j = 0; j < sizey; j++) {
for (int i = 0; i < sizex; i++) {
if (fabs(cc.rho_ave[i+sizex*j] - mc.rho_ave[i+sizex*j]) > 0.0001) {
printf("1. cell-centric and material-centric values are not equal! (%f, %f, %d, %d)\n",
cc.rho_ave[i+sizex*j], mc.rho_ave[i+sizex*j], i, j);
return false;
}

for (int mat = 0; mat < Nmats; mat++) {
if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - mc.p[ncells*mat + i+sizex*j]) > 0.0001) {
printf("2. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
cc.p[(i+sizex*j)*Nmats+mat], mc.p[ncells*mat + i+sizex*j], i, j, mat);
return false;
}

if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - mc.rho_mat_ave[ncells*mat + i+sizex*j]) > 0.0001) {
printf("3. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",
cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], mc.rho_mat_ave[ncells*mat + i+sizex*j], i, j, mat);
return false;
}
}
}
}

#ifdef DEBUG
printf("All tests passed!\n");
#endif
return true;
}
