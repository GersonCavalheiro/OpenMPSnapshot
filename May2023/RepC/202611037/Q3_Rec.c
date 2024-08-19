#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
void inicializa(int *vetor, int tam);
int encontraPrimos(int *vetor, int tam);
void crivo(int *vetor, int tam, int num);
int main(int argc, char *argv[]){
int *vetor, tam, totalPrimos;
double t_final, t_inicial;
if(argc < 3){
perror("Número de argumentos insuficiente, insira a quantidade de threads e o limite superior (n) na busca por primos.");
return 0;
}else{
omp_set_num_threads(atoi(argv[1]));
tam = atoi(argv[2])+1;
vetor = (int*) malloc(sizeof(int)*tam);
}
inicializa(vetor, tam);
t_inicial = omp_get_wtime();
totalPrimos = encontraPrimos(vetor, tam);
t_final = omp_get_wtime();
printf("Total de primos = %d\n", totalPrimos);
printf("Tempo de execução: %lf\n", t_final-t_inicial);
return 0;
}
void inicializa(int *vetor, int tam){
int i;
for(i=0;i<tam;i++){
vetor[i] = 0;
}
}
void crivo(int *vetor, int tam, int num){
int j, i;
if(num >= tam) 
return ;
#pragma omp taskloop shared(tam, num) private(j)
for(j=num+1; j < tam; j++){
if(j%num == 0){
vetor[j] = 1;
}
}
#pragma omp task private(j)
{
j=num+1;
while(vetor[j] != 0)
j++;
i = j;
crivo(vetor, tam, i);
}   
}
int encontraPrimos(int *vetor, int tam){
int i, totalPrimos = 0;
#pragma omp parallel shared(tam, vetor)
{
#pragma omp single
{
crivo(vetor, tam, 2);
}
}
for(i = 0; i < tam; i++)
if(vetor[i] == 0)
totalPrimos++;
return totalPrimos - 2; 
}