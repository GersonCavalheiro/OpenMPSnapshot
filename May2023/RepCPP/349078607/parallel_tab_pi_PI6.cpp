

#include <stdio.h>
#include <time.h>
#include <omp.h>

const long long num_steps = 100000000;
const int thread_number = 2;
const int data_spacer = 1;
double step;

int main(int argc, char* argv[])
{
omp_set_num_threads(thread_number);
clock_t start, stop;
double x, pi, sum=0.0;
volatile double tab[thread_number];
step = 1./(double)num_steps;
int i;
start = clock();
#pragma omp parallel
{
int id = omp_get_thread_num();
tab[data_spacer * id] = 0;
#pragma omp for nowait
for (i = 0; i < num_steps; i++) {
double x = (i + .5) * step;
tab[data_spacer * id] += 4.0 / (1. + x * x);
}
#pragma omp atomic
sum += tab[data_spacer * id];
}
pi = sum*step;
stop = clock();

printf("Wartosc liczby PI wynosi %15.12f\n",pi);
printf("Czas przetwarzania wynosi %f ms\n",((double)(stop - start)/1000.0));
return 0;
}