#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>
#define STEPS 100000000
#define NUM_THREADS 12
#define LOOP(i, n) for(int i = 0; i < n; i++)
void cabecalho();
void set_portuguese();
double step;
int main(int argc, char const *argv[]){
set_portuguese();
cabecalho();
printf("\nHey! Nesse script iremos calcular o valor de PI com base em calculos de integral.");
int i;
double x, pi, sum = 0.0, tempo_inicial, tempo_final;
printf("\n\nSetando o numero de Threads para %d\n", NUM_THREADS);
omp_set_num_threads(NUM_THREADS);
step = 1.0/(double) STEPS;
tempo_inicial = omp_get_wtime();
printf("\n1 - Estamos fora do contexto paralelo. Entrando...\n");
#pragma omp parallel for default(none) shared(x, step) reduction(+:sum)
LOOP(i, STEPS){
sum += 4.0 / (1.0 + pow((i + 0.5) * step, 2));
}
printf("\n2 - Estamos fora do contexto paralelo. Saindo...\n");
tempo_final = omp_get_wtime() - tempo_inicial;
printf("\nO tempo gasto no contexto paralelo foi de: %lf\n", tempo_final);
pi = sum * step;
printf("\nO valor de PI = %f\n\n", pi);
return 0;
}
void set_portuguese(){
setlocale(LC_ALL, "Portuguese");
}
void cabecalho(){
printf("\n**************************************************");
printf("\n*                                                *");
printf("\n*                                                *");
printf("\n* PROGRAMACAO PARALELA COM OPENMP - LUCCA PESSOA *");
printf("\n*                                                *");
printf("\n*                                                *");
printf("\n**************************************************\n");
}
