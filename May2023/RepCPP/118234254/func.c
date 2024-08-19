#include <stdio.h>
int jac_seq(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old) {
int i,j,k=0;
double *temp,d=10e+10;

while (k < max_iter) {
temp = u;
u = u_old;
u_old = temp;
d = 0.0;
for (i = 1; i < N-1; i++) {
for (j = 1; j < N-1; j++) {
u[i*N + j] = 0.25 * (u_old[(i-1)*N + j] + u_old[(i+1)*N + j] + u_old[i*N + (j-1)] + u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);

d += (u[i*N + j] - u_old[i*N + j]) * (u[i*N + j] - u_old[i*N + j]);
}
}
k++;
}
return(1);
}

int jac_seq_con(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old) {
int i,j,k=0;
double *temp,d=10e+10;

while (d > threshold*threshold  &&  k < max_iter) {
temp = u;
u = u_old;
u_old = temp;
d = 0.0;
for (i = 1; i < N-1; i++) {
for (j = 1; j < N-1; j++) {
u[i*N + j] = 0.25 * (u_old[(i-1)*N + j] + u_old[(i+1)*N + j] + u_old[i*N + (j-1)] + u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);

d += (u[i*N + j] - u_old[i*N + j]) * (u[i*N + j] - u_old[i*N + j]);
}
}
k++;
}
return(k);
}

int jac_mp(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old) {
int i,j,threads,k=0;
double *temp,d=10e+10;
#pragma omp parallel
{
#pragma omp single
{
threads = omp_get_num_threads();
}
}
while (k < max_iter) {
temp = u;
u = u_old;
u_old = temp;
d = 0.0;
#pragma omp parallel shared(f, u, u_old, N,threads,d) private(i, j)
{
#pragma omp for reduction(+ : d)
for (i = 1; i < N-1; i++) {
for (j = 1; j < N-1; j++) {
u[i*N + j] = 0.25 * (u_old[(i-1)*N + j] + u_old[(i+1)*N + j] + u_old[i*N + (j-1)] + u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);

d += (u[i*N + j] - u_old[i*N + j]) * (u[i*N + j] - u_old[i*N + j]);
}
}
} 
k++;
}
return(threads);
}

int jac_mp_v2(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old) {
int i,j,threads,k=0;
double *temp,d=10e+10;
#pragma omp parallel
{
#pragma omp single
{
threads = omp_get_num_threads();
}
}
#pragma omp parallel shared(f, u, u_old, N,threads,d) private(i, j) firstprivate(k)
{
while (k < max_iter) {
#pragma omp single 
{
temp = u;
u = u_old;
u_old = temp;
d = 0.0;	
}
#pragma omp for reduction(+ : d)
for (i = 1; i < N-1; i++) {
for (j = 1; j < N-1; j++) {
u[i*N + j] = 0.25 * (u_old[(i-1)*N + j] + u_old[(i+1)*N + j] + u_old[i*N + (j-1)] + u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);
d += (u[i*N + j] - u_old[i*N + j]) * (u[i*N + j] - u_old[i*N + j]);
}
}
k++;
}
} 
return(threads);
}

int jac_mp_v3(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old) {
int i,j,threads,k=0;
double *temp,d=10e+10;
#pragma omp parallel
{
#pragma omp single
{
threads = omp_get_num_threads();
}
}
#pragma omp parallel shared(f, u, u_old, N,threads,d) private(i, j) firstprivate(k)
{
while (k < max_iter) {
#pragma omp single 
{
temp = u;
u = u_old;
u_old = temp;
d = 0.0;	
}
#pragma omp for reduction(+ : d)
for (i = 1; i < N-1; i++) {
for (j = 1; j < N-1; j++) {
u[i*N + j] = 0.25 * (u_old[(i-1)*N + j] + u_old[(i+1)*N + j] + u_old[i*N + (j-1)] + u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);
d += (u[i*N + j] - u_old[i*N + j]) * (u[i*N + j] - u_old[i*N + j]);
}
}
k++;
}
} 
return(threads);
}

int jac_mp_v23(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old) {
int i,j,threads,k=0;
double *temp,d=10e+10;
#pragma omp parallel
{
#pragma omp single
{
threads = omp_get_num_threads();
}
}
#pragma omp parallel shared(f, u, u_old, N,threads,d) private(i, j) firstprivate(k)
{
while (k < max_iter) {
#pragma omp single 
{
temp = u;
u = u_old;
u_old = temp;
d = 0.0;	
}
#pragma omp for reduction(+ : d)
for (i = 1; i < N-1; i++) {
for (j = 1; j < N-1; j++) {
u[i*N + j] = 0.25 * (u_old[(i-1)*N + j] + u_old[(i+1)*N + j] + u_old[i*N + (j-1)] + u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);
d += (u[i*N + j] - u_old[i*N + j]) * (u[i*N + j] - u_old[i*N + j]);
}
}
k++;
}
} 
return(threads);
}

int gauss(int N, double delta, double threshold, int max_iter, double *f, double *u) {
int k=0;
double u_ij,d=10e+10;
while (k < max_iter) {
d = 0.0;
for (int i = 1; i < N-1; i++) {
for (int j = 1; j < N-1; j++) {
u_ij = u[i*N + j];
u[i*N + j] = 0.25 * (u[(i-1)*N + j] + u[(i+1)*N + j] + u[i*N + (j-1)] + u[i*N + (j+1)] + delta*delta*f[i*N + j]);

d += (u[i*N + j] - u_ij) * (u[i*N + j] - u_ij);
}
}
k++;
}
return(1);
}

int gauss_con(int N, double delta, double threshold, int max_iter, double *f, double *u) {
int k=0;
double u_ij,d=10e+10;
while (d > threshold*threshold  &&  k < max_iter) {
d = 0.0;
for (int i = 1; i < N-1; i++) {
for (int j = 1; j < N-1; j++) {
u_ij = u[i*N + j];
u[i*N + j] = 0.25 * (u[(i-1)*N + j] + u[(i+1)*N + j] + u[i*N + (j-1)] + u[i*N + (j+1)] + delta*delta*f[i*N + j]);

d += (u[i*N + j] - u_ij) * (u[i*N + j] - u_ij);
}
}
k++;
}
return(k);
}

int mandel(int disp_width, int disp_height, int *array, int max_iter) {
double 	x, y, u, v, u2, v2, scale_real, scale_imag;
int 	i, j, iter, threads;

scale_real = 3.5 / (double)disp_width;
scale_imag = 3.5 / (double)disp_height;

#pragma omp parallel
{
#pragma omp single
{
threads = omp_get_num_threads();
}
}

#pragma omp parallel shared(array,disp_width,disp_height,scale_real,scale_imag,max_iter) private(x, y, u, v, u2, v2, i, j, iter)
{
#pragma omp for
for(i = 0; i < disp_width; i++) {
x = ((double)i * scale_real) - 2.25; 
for(j = 0; j < disp_height; j++) {
y = ((double)j * scale_imag) - 1.75; 
u    = 0.0;
v    = 0.0;
u2   = 0.0;
v2   = 0.0;
iter = 0;
while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
v = 2 * v * u + y;
u = u2 - v2 + x;
u2 = u*u;
v2 = v*v;
iter = iter + 1;
}
iter = iter == max_iter ? 0 : iter;

array[i*disp_height + j] = iter;
}
}
}
return(threads);
}

void write_result(double *U, int N, double delta, char filename[20]) {
double u, y, x;
FILE *matrix=fopen(filename, "w");
for (int i = 0; i < N; i++) {
x = -1.0 + i * delta + delta * 0.5;
for (int j = 0; j < N; j++) {
y = -1.0 + j * delta + delta * 0.5;
u = U[i*N + j];
fprintf(matrix, "%g\t%g\t%g\n", x,y,u);
}
}
fclose(matrix);
}
