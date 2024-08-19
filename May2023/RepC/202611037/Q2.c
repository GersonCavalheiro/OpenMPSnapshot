#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
void uniqueRandom(int *vetor, int tam); 
void rankSort(int **vetor, int tam);
int main(int argc, char *argv[]){
int *vetor,  tam;
double t_final, t_inicial;
srand(time(NULL));
if(argc < 3){
perror("Número de argumentos insuficiente, insira a quantidade de threads e o tamanho do vetor respectivamente.");
return 0;
}else{
omp_set_num_threads(atoi(argv[1]));
tam = atoi(argv[2]);
vetor = (int*) malloc(sizeof(int)*tam);
}
puts("Começando a gerar randômicos únicos");
uniqueRandom(vetor, tam);
puts("Randômicos gerados");
t_inicial = omp_get_wtime();
rankSort(&vetor, tam);
t_final = omp_get_wtime();
printf("Tempo de execução: %lf\n", t_final-t_inicial);
return 0;
}
void uniqueRandom(int *vetor, int tam){
int i = 0, *aux, random; 
aux = (int*) malloc(sizeof(int)*tam);
for(i=0; i<tam; i++){
aux[0]=0;
}
i= 0;
while(i < tam){
random = rand()%tam;
if(aux[random] == 0){
vetor[i] = random;
aux[random] = 1;
i++;
}
}
}
void rankSort(int **vetor, int tam){
int rank=0, i, j, *ordenado;
ordenado = (int*) malloc(sizeof(int)*tam);
#pragma omp parallel for private(i,j) firstprivate(rank) shared(vetor, ordenado, tam)
for(i = 0; i<tam; i++){
for(j=0; j<tam; j++){
if((*vetor)[i] > (*vetor)[j])
rank++;
}
ordenado[rank] = (*vetor)[i];
rank = 0;
}
free(*vetor);
*vetor = ordenado;
}