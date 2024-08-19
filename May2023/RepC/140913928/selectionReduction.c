

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


struct Compara{
int valor;
int indice;
};


#pragma omp declare reduction(min : struct Compara : omp_out = omp_in.valor > omp_out.valor ? omp_out : omp_in)
void selectionsort(int *vet, int tam);
void exibevetor(int *vet, int tam);


void selectionsort(int *vet, int tam){   
int i,j,aux;
struct Compara menor;

#pragma omp simd reduction(min:menor)  
for (i = 0; i < (tam - 1); ++i){ 
menor.valor = vet[i];
menor.indice = i;  
for (j = i + 1; j < tam; ++j){
if (vet[j] < menor.valor){ 
menor.valor = vet[j];
menor.indice = j; 
}
}

aux = vet[i];
vet[i] = menor.valor;
vet[menor.indice] = aux;
}
}



void exibevetor(int *vet, int tam){
int i;
for (i = 0; i < tam; ++i){
printf("%d ", vet[i]);
}
printf("\n");
}

int main(){
int *vet, i, tam;
clock_t t, end;
double cpu_time_used;

printf("Digite o tamanho do vetor:\n");
scanf("%d",&tam);
vet = (int *)malloc(sizeof(int)*tam);
if(vet == NULL){
exit(1);
}

for(i = 0; i < tam; ++i){
vet[i] = rand() % 100; 
}

t = clock();
exibevetor(vet,tam);
selectionsort(vet, tam);
t = clock()-t;
exibevetor(vet,tam);
cpu_time_used = ((double)t)/CLOCKS_PER_SEC;
printf("\nTempo de execução: %f\n", cpu_time_used);
free(vet);

return 0;
}