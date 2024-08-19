#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "timer.h"
int main(int argc, char **argv)
{
const long N = 10000;
const long M = 1000000;
double tempo, fim, inicio;
int i, j, cont, T;
T = atoi(argv[1]);
GET_TIME(inicio);
for (i = 0; i < N; i++)
#pragma omp parallel for num_threads(T) private(i,j,cont)
for(j = 0; j < M; j++)
cont = cont + 1;
GET_TIME(fim);
tempo = fim - inicio;
printf("Tempo: %.8lf\n", tempo);
return 0;
}
