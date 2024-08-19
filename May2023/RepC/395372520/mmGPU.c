#include <stdio.h>
#include <stdlib.h>
void mm(double *a, double *b, double *c, int width)
{
#pragma omp target map(to                                                       : a [0:(width * width)], b [0:(width * width)]) map(from : c [0:(width * width)])
#pragma omp teams distribute parallel for
for (int i = 0; i < width; i++)
{
for (int j = 0; j < width; j++)
{
double sum = 0;
for (int k = 0; k < width; k++)
{
double x = a[i * width + k];
double y = b[k * width + j];
sum += x * y;
}
c[i * width + j] = sum;
}
}
}
int main()
{
int width = 2000;
double *a = (double *)malloc(width * width * sizeof(double));
double *b = (double *)malloc(width * width * sizeof(double));
double *c = (double *)malloc(width * width * sizeof(double));
#pragma omp parallel for collapse(2)
for (int i = 0; i < width; i++)
{
for (int j = 0; j < width; j++)
{
a[i * width + j] = i;
b[i * width + j] = j;
c[i * width + j] = 0;
}
}
mm(a, b, c, width);
}