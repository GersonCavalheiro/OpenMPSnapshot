#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h> 
int main(int argc, char* argv[]) {
int nthreads, tid; 
#pragma omp parallel 
{ 
tid = omp_get_thread_num(); 
printf("Hola, soy el hilo: = %d\n",tid); 
if(tid == 0){ 
nthreads = omp_get_num_threads();
printf("Número de hilos: %d\n",nthreads);
}
}
return 0;
} 
