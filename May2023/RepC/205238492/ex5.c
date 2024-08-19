#include <stdio.h>
#include <omp.h>
#include "timer.h"
int main(int argc, char **argv)
{
const long N = 10000;
const long M = 100000;
double tempo, fim, inicio;
int i, j, cont;
GET_TIME(inicio);
#pragma omp parallel for
for (i = 0; i < N; i++)
for(j = 0; j < M; j++)
cont = cont + 1;
GET_TIME(fim);
tempo = fim - inicio;
printf("Tempo: %.8lf\n", tempo);
return 0;
}
