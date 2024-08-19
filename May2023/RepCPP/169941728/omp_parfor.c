#include <omp.h>
#include <stdio.h>
#include <time.h>

double CLOCK()
{
struct timespec t;
clock_gettime(CLOCK_MONOTONIC, &t);
return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

main(int argc, char **argv)
{
unsigned int i;
double start, finish, total1, total2, total3;
double a[1000000];

start = CLOCK();
for (i = 0; i < 1000000; i++)
{
a[i] = 2.0 * i;
a[i] += (i % 3);
}
finish = CLOCK();
total1 = finish - start;

start = CLOCK();
#pragma omp parallel for
for (i = 0; i < 1000000; i++)
{
a[i] = 2.0 * i;
a[i] += (i % 3);
}
finish = CLOCK();
total2 = finish - start;

start = CLOCK();
for (i = 0; i < 1000000; i++)
{
a[i] = 2.0 * i;
a[i] += (i % 3);
}
finish = CLOCK();
total3 = finish - start;


printf("Time for first loop = %f\n", total1);
printf("Time for second loop = %f\n", total2);
printf("Time for third loop = %f\n", total3);

return 0;
}
