#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include "drand.h"
#include "timing.h"
int main(int argc, char const *argv[]) {
int threads = (int) strtoimax(argv[1], NULL, 10);
omp_set_num_threads(threads);
printf("threads = %d\n", threads);
dsrand(12345);
double wcs, wce, ct;
timing(&wcs, &ct);
int N0 = 0, N1 = 0;
#pragma omp parallel default(none) shared(N0, N1)
{
#pragma omp single
{
for (int x = -2000; x < 500; x++) {
for (int y = 0; y < 1250; y++) { 
#pragma omp task
{
double cr = (drand() + (double)x) * 0.001;
double ci = (drand() + (double)y) * 0.001;
double zr = cr;
double zi = ci;
int i = 0;
for (; i < 20000; i++) {
double new_zr = zr * zr - zi * zi + cr;
double new_zi = 2 * zr * zi + ci;
zr = new_zr;
zi = new_zi;
if (zr * zr + zi * zi > 4) break;
}
if (i == 20000) {
#pragma omp critical(N1)
N1++; 
} else {
#pragma omp critical(N0)
N0++; 
}
}
}
}
}
}
timing(&wce, &ct);
double runtime = wce - wcs;
double area = 2 * N1 * 3.125 / (N1 + N0);
printf("area = %f, ", area);
printf("runtime = %f\n\n", runtime);
return 0;
}
