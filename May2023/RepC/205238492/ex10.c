#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "timer.h"
int main(int argc, char const *argv[])
{
unsigned long long N;
unsigned long long POS;
unsigned long long VAL;
long max_val = 0;
long i;
long *vet;
int T;
double tempo, fim, inicio;
N = atoll(argv[1]);
T = atoi(argv[2]);
VAL = (argc > 3) ? atoll(argv[3]) : 2999999999;
POS = (argc > 4) ? atoll(argv[4]) : 100;
vet = malloc(N * sizeof(*vet));
GET_TIME(inicio);
#pragma omp parallel for num_threads(T)
for (i = 0; i < N; i++) {
vet[i] = i;
}
vet[POS] = VAL;
GET_TIME(fim);
tempo = fim - inicio;
printf("Tempo: %.8lf\n", tempo);
GET_TIME(inicio);
#pragma omp parallel for reduction(max:max_val) num_threads(T)
for (i = 0; i < N; i++) {
if (vet[i] > max_val)
max_val = vet[i];
}
GET_TIME(fim);
tempo = fim - inicio;
printf("Tempo: %.8lf\n", tempo);
printf("\nMax val = %ld\n", max_val);
free(vet);
return 0;
}
