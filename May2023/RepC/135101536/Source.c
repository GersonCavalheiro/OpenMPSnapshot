#include <stdio.h>
#include <omp.h>
#define ARRAY_MAX 2000000
int main()
{
int a[ARRAY_MAX], i;
double before, after, sub;
before = omp_get_wtime();
#pragma omp parallel for
for (i = 0; i < ARRAY_MAX; ++i) {
a[i] = 0;
}
#pragma omp parallel for
for (i = 0; i < ARRAY_MAX; ++i) {
a[i] += i;
}
after = omp_get_wtime();
sub = after - before;
printf("%f\n%f\n%f\n", before, after, sub);
return 0;
}