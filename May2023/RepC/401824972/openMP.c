#include "openMP.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define NTHREADS 4


int eratostenesOpenMP(int lastNumber){

char* isPrime = malloc(lastNumber+1 * sizeof(int));

for (int i = 0; i <= lastNumber; i++){
isPrime[i] = 1;
}


#pragma omp parallel num_threads(NTHREADS){
#pragma omp single
printf("Inicio da região paralela \n Número de threads = %d \n",  omp_get_num_threads()); 

#pragma omp for
for (int i = 2; i*i <= lastNumber; i++){
if (isPrime[i]){
for (int j = i*i; j <= lastNumber; j += i){
isPrime[j] = 0;
}          
}

printf("Thread %d executa interação %d do for \n", omp_get_thread_num(),i);
}
}

int contador = 0;

for (int i = 2; i <= lastNumber; i++){
if(isPrime[i]){
contador = contador + 1;
}
}

free(isPrime);
return contador;

}
