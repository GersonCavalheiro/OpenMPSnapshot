#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
int main(int argc, char *argv[])
{
double a = 0.0;
double b = 10.0;
double exact = 0.49936338107645674464;
int n = 1000000;
double pi = 3.141592653589793;
printf("\n");
printf("QUAD:\n");
printf("  C version\n");
printf("\n");
printf("  Estimate the integral of f(x) from A to B.\n");
printf("  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n");
printf("  A = %f\n", a);
printf("  B = %f\n", b);
printf("  Exact integral from 0 to 10 is 0.49936338107645674464...\n");
double *x = (double *)malloc(n * sizeof(double));
for (int i = 0; i < n; i++)
{
x[i] = ((double)(n - i - 1) * a + (double)(i)*b) / (double)(n - 1);
}
double wtime1 = omp_get_wtime();
double total = 0.0;
#pragma omp parallel shared(n, pi, x)
#pragma omp for reduction(+ : total)
for (int i = 0; i < n; i++)
{
total = total + 50 / pi / (2500.0 * x[i] * x[i] + 1.0);
}
int flops = 6 * n;
double wtime2 = omp_get_wtime();
total = (b - a) * total / (double)n;
double error = fabs(total - exact);
double wtime = wtime2 - wtime1;
double mflops = (double)(flops) / 1000000.0 / wtime;
printf("\n");
printf("  Estimate = %f\n", total);
printf("  Error    = %e\n", error);
printf("  W time   = %f\n", wtime);
printf("  FLOPS    = %d\n", flops);
printf("  MFLOPS   = %f\n", mflops);
free(x);
return 0;
}
