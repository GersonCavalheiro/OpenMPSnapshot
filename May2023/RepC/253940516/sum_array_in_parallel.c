#include <omp.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>
#define MAX 100
#define LOOP(i, n) for(int i = 0; i < n; i++)
void cabecalho();
void set_portuguese();
int main(int argc, char const *argv[]){
set_portuguese();
cabecalho();
int i;
int *A, *B, *C;
printf("\nAlocando e Inicializando os Arrays...");
A = (int *)malloc(MAX*sizeof(int));
B = (int *)malloc(MAX*sizeof(int));
C = (int *)malloc(MAX*sizeof(int));
printf("\n\nPreenchendo os Arrays com os dados a serem somados");
LOOP(i, MAX){
A[i] = i * 2;
B[i] = i * 3;
}
printf("\n\nExibindo valores dos Arrays A e B...\n\n");
LOOP(i, MAX){
printf("\t%d \t %d\n", A[i], B[i]);
}
printf("\nEstamos fora do contextos paralelo... Iremos realizar a soma dos vetores em paralelo...\n");
#pragma omp parallel for default(none) shared(A, B, C)
LOOP(i, MAX){
C[i] = A[i] + B[i];
}
printf("\nProcessamos a soma e saímos da região paralela...");
printf("\n\nExibindo valores da soma dos Arrays...\n\n");
LOOP(i, MAX){
printf("\t%d\n", C[i]);
}
printf("\nFim do programa... Iremos dar um free nos Arrays alocados...\n\n");
free(A);
free(B);
free(C);
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
