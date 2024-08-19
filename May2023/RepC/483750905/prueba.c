#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define ITERACIONES 2100000000
#define SECUENCIAL 0
void showTimeSeq();
void showTimePrl();
void main(){
omp_set_num_threads(omp_get_num_procs()); 
float tiempo_inicial= omp_get_wtime();
if (SECUENCIAL == 1) 
showTimeSeq();
else
showTimePrl();
float tiempo_final= omp_get_wtime();
printf("%lf \n",tiempo_inicial);
printf("%lf \n",tiempo_final);
printf("%lf \n %i",tiempo_final-tiempo_inicial,SECUENCIAL);
}
void showTimeSeq(){
for(int i = 0; i< ITERACIONES ;i++){
int a =0;
}
}
void showTimePrl(){
#pragma omp parallel
for(int i = 0; i< (ITERACIONES/2) ;i++){
int a =0;
}
}
