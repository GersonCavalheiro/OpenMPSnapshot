

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef struct bucket{
int *vet;
int tam;
int min;
int max;
}Balde;



void insereNoBalde(Balde *balde, int elem){
balde->vet[balde->tam++] = elem;
}


void insertionSort(int *vet, int tam){
int i, j, aux;
for (i = 0; i < tam; ++i){
aux = vet[i];
j = i - 1;
while(j >= 0 && vet[j] > aux){
vet[j+1] = vet[j];
j = j - 1; 
}
vet[j+1] = aux;
}
}



int* geraVetor(int tam, int* min, int* max) {
int *vet = (int *)malloc(sizeof(int) * tam);
int i;

srand(time(NULL));
vet[0] = rand() % RAND_MAX;
*min = vet[0];
*max = vet[0];

for (i = 1; i < tam; ++i) {
vet[i] = rand() % RAND_MAX;
if (*max < vet[i]){
*max = vet[i];
}
if (*min > vet[i]){
*min = vet[i];
}
}
return vet;
}



void exibevetor(int *vet, int tam){
int i;
for(i = 0; i < tam; ++i){
printf("%d ", vet[i]);
}
printf("\n");
}

int main(int argc, char const *argv[])
{
int i,j,min,max;
clock_t t, end;
double cpu_time_used;
if (argc != 3) {
fprintf(stderr, "Digite: [tamanho do vetor] [numero de threads]\n");
exit(1);
}

int tam = atoi(argv[1]); 
int numThreads = atoi(argv[2]);  
int *vet1 = geraVetor(tam,&min,&max); 

Balde balde[numThreads];

int range = (int) ceil((float)(max - min) / (float)(numThreads-1));
balde[0].min = min;
balde[0].max = min + range;
balde[0].vet = (int*) malloc(sizeof(int) * tam);
balde[0].tam = 0;


for(i = 1; i < numThreads - 1; ++i) {
balde[i].min = balde[i - 1].max + 1;
balde[i].max = balde[i].min + range;
balde[i].vet = (int*) malloc(sizeof(int) * tam);
balde[i].tam = 0;
}

for(i = 0; i < tam; ++i){

#pragma omp parallel for num_threads(numThreads)
for (j = 0; j < numThreads - 1; ++j) {
if(vet1[i] <= balde[j].max && vet1[i] >= balde[j].min)
insereNoBalde(&balde[j], vet1[i]); 
}
}

for (i = 0; i < numThreads; ++i) {
if(balde[i].tam!=0){ 
insertionSort(balde[i].vet, balde[i].tam); 
}
}

int cont = 0;

for (i = 0; i < numThreads; ++i) {
for (j = 0; j < balde[i].tam; ++j) {
vet1[cont++] = balde[i].vet[j]; 
}
}

exibevetor(vet1,tam); 

t = clock(); 

cpu_time_used = ((double)t)/CLOCKS_PER_SEC;
printf("\nTempo de execução: %f\n", cpu_time_used);

return 0;
}


