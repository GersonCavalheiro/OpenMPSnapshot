#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define TAM 100
#define NUM_THREADS 10
void imprimeVetor(int *vet,int tam){
int i;
for(i=0;i<tam;i++){
printf("%d ", vet[i]);
}
printf("\n");
}
void merge(int *vet,int inicio,int meio,int fim){
int *aux, i, j, k,tam,t,fim1,fim2;
i = inicio;
j = meio+1;
fim1 = 0;
fim2 = 0;
tam = fim - inicio + 1;
aux = (int *)malloc(sizeof(int)*TAM);
for(k = 0; k < tam; k++){
if( (!fim1) && (!fim2) ){
if(vet[i] < vet[j]){
aux[k] = vet[i];
i++;
}
else{
aux[k] = vet[j];
j++;  
}
if(i > meio)
fim1 = 1;
if(j > fim)
fim2 = 1;
}else{
if(fim1 != 1){
aux[k] = vet[i];
i++;
}else{
aux[k] = vet[j];
j++;
}
}
}
for(k = 0; k < tam; k++){
vet[inicio+k] = aux[k];
}
free(aux);
}
void mergeSerial(int *vet,int inicio,int fim){
int meio;
if(inicio < fim){
meio = floor(inicio+fim)/2;
mergeSerial(vet,inicio,meio);
mergeSerial(vet,meio+1,fim);
merge(vet,inicio,meio,fim);
}
}
void mergeSort(int *vet,int inicio,int fim,int threads){
int meio;
if(threads==1){
mergeSerial(vet,inicio,fim);
}
else{
meio = floor(inicio+fim)/2;
#pragma omp parallel sections num_threads(NUM_THREADS)
{
#pragma omp section
mergeSort(vet,inicio,meio,threads/2);
#pragma omp section
mergeSort(vet,meio+1,fim,threads-threads/2);
} 
merge(vet,inicio,meio,fim);
}
}
int main(){
int *vet,i,num_threads;
double inicio,fim,tempo;
vet = (int *)malloc(sizeof(int)*TAM);
if(vet==NULL){
exit(1);
}
#pragma omp parallel
{
#pragma omp master
{
num_threads = omp_get_num_threads();
}
}
printf("Numero de threads: %d\n",num_threads);
srand(time(NULL));
for(i=0;i<TAM;i++){
vet[i] = rand()%TAM; 
}
imprimeVetor(vet,TAM);  
inicio = omp_get_wtime();
mergeSort(vet,0,TAM-1,num_threads);
imprimeVetor(vet,TAM);
fim = omp_get_wtime();
printf("Função demorou %f segundos para ordenar\n", fim-inicio);
free(vet);
return 0;
}