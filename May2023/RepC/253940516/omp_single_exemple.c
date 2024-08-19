#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#define NUM_THREADS 12
void cabecalho();
void set_portuguese();
int main(int argc, char const *argv[]){
set_portuguese();
cabecalho();
int thread_id;
printf("\n1 - Estamos fora do contexto paralelo. Entrando...\n\n");
#pragma omp parallel num_threads(2)
{
#pragma omp single
printf("Read input\n");
printf("Compute results\n");
#pragma omp single
printf("Write output\n");
}
printf("\n2 - Estamos fora do contexto paralelo. Saindo...\n");
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
