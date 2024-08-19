

#include <stdio.h>
#include <time.h>
#include <omp.h>

long long num_steps = 100000000;
double step;

int main(int argc, char* argv[])
{
omp_set_num_threads(2);
clock_t start, stop;
double x, pi, sum=0.0;
double tab[10 * 8];
step = 1./(double)num_steps;
int i;
start = clock();
#pragma omp parallel
{
int id = omp_get_thread_num();
tab[10 * id] = 0;
#pragma omp for nowait
for (i = 0; i < num_steps; i++) {
double x = (i + .5) * step;
tab[10 * id] += 4.0 / (1. + x * x);
}
#pragma omp atomic
sum += tab[10 * id];
}
pi = sum*step;
stop = clock();

printf("Wartosc liczby PI wynosi %15.12f\n",pi);
printf("Czas przetwarzania wynosi %f ms\n",((double)(stop - start)/1000.0));
return 0;
}