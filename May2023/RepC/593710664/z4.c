#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char* argv[])
{
double n;
double sum1 = 0;
double sum2 = 0;
printf("Number: ");
scanf("%lf", &n);
int tc = n;
double s1 = omp_get_wtime();
#pragma omp parallel for num_threads(tc) reduction(+: sum1)
for (int i = 1; i <= (int)n; i++)
sum1 += (double)i;
s1 = omp_get_wtime() - s1;
printf("\nSum is %lf and executed for %lfs with reduction\n", sum1, s1);
double s2 = omp_get_wtime();
#pragma omp parallel for
for (int i = 1; i <= (int)n; ++i)
{
sum2 += i;
}
s2 = omp_get_wtime() - s2;
printf("\nSum is %lf and executed for %lfs without reduction\n", sum2, s2);
return 0;
}
