#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
void initClock(struct timeval *begin) {
gettimeofday(begin, 0);
}
void endClock(struct timeval *end) {
gettimeofday(end, 0);
}
void getElapsedTime(struct timeval *begin, struct timeval *end) {
long seconds = end->tv_sec - begin->tv_sec;
long microseconds = end->tv_usec - begin->tv_usec;
double elapsed = seconds + microseconds*1e-6;
printf("[TIME] %.3f sec.\n", elapsed);
}
void algSecuencial(long unsigned int iterations) {
double respuesta = 0.0;
long unsigned int numeroIteraciones = iterations;
for(long indice = 0; indice <= numeroIteraciones; indice++){
if(indice % 2 == 0){
respuesta += 4.0 / (2.0 * indice + 1.0);
}else{
respuesta -= 4.0 / (2.0 * indice + 1.0);
}
}
printf("[SECUENCIAL] %d | %.8f\n", numeroIteraciones, respuesta);
}
void algParalelo(long unsigned int iterations) {
int numeroHilos = 4, idHilo;
omp_set_num_threads(numeroHilos);
double respuesta = 0.0, sumasParciales[numeroHilos];
long unsigned int numeroIteraciones = iterations;
#pragma omp parallel private(idHilo) shared(sumasParciales)
{
int idHilo = omp_get_thread_num();
sumasParciales[idHilo] = 0.0;
for(long indice = idHilo; indice < numeroIteraciones; indice += numeroHilos) {
if(indice % 2 == 0) {
sumasParciales[idHilo] += 4.0 / (2.0 * indice + 1.0);
} else {
sumasParciales[idHilo] -= 4.0 / (2.0 * indice + 1.0);
}
}
}
for(int indice = 0; indice < numeroHilos; indice++) {
respuesta += sumasParciales[indice];
}
printf("[PARALELO  ] %d | %.8f\n", numeroIteraciones, respuesta);
}
int main() {
struct timeval begin, end;
long unsigned int iterations;
printf("Numero de iteracciones: ");
scanf("%lud", &iterations);
printf("\n");
for(int i = 0; i < 10; i++) {
initClock(&begin);
algSecuencial(iterations);
endClock(&end);
getElapsedTime(&begin, &end);
printf("\n"); 
initClock(&begin);
algParalelo(iterations);
endClock(&end);
getElapsedTime(&begin, &end);
long unsigned int increment;
if(i == 0 || i % 2 == 0) {
increment = iterations / 2;
}
iterations = iterations + increment;
printf("\n\n"); 
}
return 0;
}