#include <stdio.h>  
#include <omp.h>  
#include <unistd.h>
int main() { 
printf("Fora = %d\n", omp_in_parallel()); 
omp_set_num_threads(6);
#pragma omp parallel num_threads(4) default(none) 
{  
int i = omp_get_thread_num(); 
printf("Olá da thread %d\n", i);
printf("Dentro = %d\n", omp_in_parallel( )); 
printf("Número de threads = %d\n",omp_get_num_threads());
}  
}
