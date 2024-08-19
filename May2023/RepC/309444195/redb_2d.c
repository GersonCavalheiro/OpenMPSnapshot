#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#define  N   (64+2)
float maxeps = 0.1e-7;
int itmax = 100;
float w = 0.5;
float eps;

float A[N][N];

void relax();
void init();
void verify();
int run_read_black_2d();

int main(int an, char **as) {
int status;
struct timeval start, stop;
double secs = 0;

gettimeofday(&start, NULL);
status = run_read_black_2d();
gettimeofday(&stop, NULL);

secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
printf("time taken for thread=%d, N=%d: %f seconds\n", omp_get_max_threads(), N, secs);
return status;
}

int run_read_black_2d(){
int it;
init();

for (it = 1; it <= itmax; it++) {
eps = 0.;
relax();
if (eps < maxeps) break;
}

verify();

return 0;
}


void init() {
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i <= N - 1; i++)
for (int j = 0; j <= N - 1; j++){
if (i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
else A[i][j] = (1. + i + j);
}
}


void relax() {

#pragma omp parallel for  schedule(static) collapse(2) reduction(max : eps)
for (int i = 1; i <= N - 2; i++)
for (int j = 1; j <= N - 2; j++){
if ((i + j) % 2 == 1) {
float b;
b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
eps = Max(fabs(b), eps);
A[i][j] = A[i][j] + b;
}}

#pragma omp parallel for collapse(2) schedule(static)
for (int i = 1; i <= N - 2; i++)
for (int j = 1; j <= N - 2; j++)
if ((i + j) % 2 == 0) {
float b;
b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
A[i][j] = A[i][j] + b;
}

}


void verify() {
float s;

s = 0.;
#pragma omp parallel for  schedule(static) collapse(2) reduction (+: s)
for (int i = 0; i <= N - 1; i++)
for (int j = 0; j <= N - 1; j++){
s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
}
printf("  S = %f\n", s);
}
