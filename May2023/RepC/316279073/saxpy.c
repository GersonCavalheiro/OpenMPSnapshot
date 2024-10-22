#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define TWO02 (1 <<  2)
#define TWO04 (1 <<  4)
#define TWO08 (1 <<  8)
#define TWO27 (1 << 27)
int main(int argc, char *argv[])
{
int   i, n = TWO27,
iret = 0;
float a = 101.0f / TWO02,
*x, *y, *z;
struct timespec rt[2];
double wt; 
if (NULL == (x = (float *) malloc(sizeof(*x) * n))) {
printf("error: memory allocation for 'x'\n");
iret = -1;
}
if (NULL == (y = (float *) malloc(sizeof(*y) * n))) {
printf("error: memory allocation for 'y'\n");
iret = -1;
}
if (NULL == (z = (float *) malloc(sizeof(*z) * n))) {
printf("error: memory allocation for 'z'\n");
iret = -1;
}
if (0 != iret) {
free(x);
free(y);
free(z);
exit(EXIT_FAILURE);
}
for (i = 0; i < n; i++) {
x[i] =        rand() % TWO04 / (float) TWO02;
y[i] = z[i] = rand() % TWO08 / (float) TWO04;
}
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp parallel default(none) shared(n, a, x, y) private(i)
{
#pragma omp for simd schedule(simd:static)
for (i = 0; i < n; i++) {
y[i] = a * x[i] + y[i];
}
}
clock_gettime(CLOCK_REALTIME, rt + 1);
wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
printf("saxpy on host : %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n / (1.0e6 * wt));
clock_gettime(CLOCK_REALTIME, rt + 0);
#pragma omp target device(0) map(to:n, a, x[0:n]) map(tofrom:z[0:n]) private(i)
{
for (i = 0; i < n; i++) {
z[i] = a * x[i] + z[i];
}
}
clock_gettime(CLOCK_REALTIME, rt + 1);
wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
printf("saxpy on accel: %9.3f sec %9.1f MFLOPS\n", wt, 2.0 * n / (1.0e6 * wt));
for (i = 0; i < n; i++) {
iret = *(int *) (y + i) ^ *(int *) (z + i);
assert(iret == 0);
}
return 0;
}
