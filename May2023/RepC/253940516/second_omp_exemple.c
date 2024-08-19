#include <omp.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>
#define NAME_SIZE 256
void cabecalho();
void set_portuguese();
int main(int argc, char const *argv[]){
set_portuguese();
cabecalho();
int thread_id, num_threads;
char *name = malloc(NAME_SIZE);
if (name == NULL){
printf("Sorry... We dont have memory :(\n");
return 1;
}
printf("\nHey coder! What's your name? ");
scanf("%[^\n]s",name);
printf("\nHello %s. Nice to meet you.\n", name);
printf("\n1 - We are out of the parallel context.\n\n");
#pragma omp parallel
{
thread_id = omp_get_thread_num();
num_threads = omp_get_num_threads();
printf("Hey %s! I'm Thread %d - Total %d!\n", name, thread_id, num_threads);
}
printf("\n2 - We are out of the parallel context.\n\n");
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
