

#include <stdio.h>
#include <time.h>
#include <omp.h>

const long long num_steps = 100000000;
const int thread_number = 2;
double step;

int main(int argc, char* argv[])
{
omp_set_num_threads(thread_number);
double start, stop;
double x, pi, sum=0.0;
step = 1./(double)num_steps;
int i;
for (int data_spacer = 0; data_spacer < 30; data_spacer++) {
volatile double tab[data_spacer + 1] = {0};
pi = 0;
sum = 0;
start = omp_get_wtime();
#pragma omp parallel
{
int id = omp_get_thread_num();
tab[data_spacer + id] = 0;
#pragma omp for nowait
for (i = 0; i < num_steps; i++) {
double x = (i + .5) * step;
tab[data_spacer + id] += 4.0 / (1. + x * x);
}
#pragma omp atomic
sum += tab[data_spacer + id];
}
pi = sum * step;
stop = omp_get_wtime();

printf("Czas przetwarzania wynosi %f ms dla odleglosci %d\n", (stop - start) * 1000, data_spacer);
}
return 0;
}