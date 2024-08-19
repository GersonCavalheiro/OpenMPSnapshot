#include <omp.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>
void cabecalho();
void set_portuguese();
int main(int argc, char const *argv[]){
set_portuguese();
cabecalho();
printf("\nNum de processadores disponivel no momento: %d", omp_get_num_procs());
printf("\n\n1 - Entrando no contexto paralelo...\n");
#pragma omp parallel
{
#pragma omp master
{
printf("\nNum de processadores disponivel no momento: %d", omp_get_num_procs());
}
}
printf("\n\n2 - Saindo do contexto paralelo...\n\n");
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
