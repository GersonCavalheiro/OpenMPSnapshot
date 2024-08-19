

#include <omp.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#include "pngwriter.h"

const int MAX_ITER_COUNT = 512;
const int DIFF_ITER_COUNT = -1;
const int MAX_DEPTH = 6;
const int MIN_SIZE = 32;
const int SUBDIV = 4;


float abs2(complex v)
{
return creal(v) * creal(v) + cimag(v) * cimag(v);
}

int kernel(int w, int h, complex cmin, complex cmax,
int x, int y)
{
complex dc = cmax - cmin;
float fx = (float)x / w;
float fy = (float)y / h;
complex c = cmin + fx * creal(dc) + fy * cimag(dc) * I;
int iteration = 0;
complex z = c;
while (iteration < MAX_ITER_COUNT && abs2(z) < 2 * 2) {
z = z * z + c;
iteration++;
}
return iteration;
}


void mandelbrot_block(int *iter_counts, int w, int h, complex cmin,
complex cmax, int x0, int y0, int d)
{
#pragma omp parallel for schedule(guided)
for (int i = x0; i < x0 + d; i++) {
for (int j = y0; j < y0 + d; j++) {
iter_counts[j * w + i] = kernel(w, h, cmin, cmax, i, j);
}
}
}


int main(int argc, char **argv)
{
const int w = 2048;
const int h = w;
int *iter_counts;

complex cmin, cmax;

int pic_bytes = w * h * sizeof(int);
iter_counts = (int *)malloc(pic_bytes);

cmin = -1.5 + -1.0 * I;
cmax = 0.5 + 1.0 * I;

double t1 = omp_get_wtime();
mandelbrot_block(iter_counts, w, h, cmin, cmax, 0, 0, w);
double t2 = omp_get_wtime();

save_png(iter_counts, w, h, "mandelbrot.png");

double walltime = t2 - t1;
printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n",
walltime, h * w * 1e-6 / walltime);

free(iter_counts);
return 0;
}

