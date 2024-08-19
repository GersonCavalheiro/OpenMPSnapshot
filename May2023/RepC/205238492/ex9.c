#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "timer.h"
int main(int argc, char const *argv[])
{
unsigned long long num_steps;
int T;
int i;
double step;
double x;
double pi;
double sum = 0.0;
double tempo, fim, inicio;
num_steps = atoll(argv[1]);
T = atoi(argv[2]);
printf("Num steps=%lld\nT=%d\n", num_steps, T);
step = 1.0/(double) num_steps;
GET_TIME(inicio);
#pragma omp parallel for reduction(+:sum) num_threads(T) private(i,x)
for (i = 0; i < num_steps; i++) {
x = (i + 0.5) * step;
sum = sum + 4.0/(1.0 + x*x);
}
pi = step * sum;
GET_TIME(fim);
tempo = fim - inicio;
printf("Tempo: %.8lf\n", tempo);
printf("\n\nPI: %.20lf\n", pi);
return 0;
}
