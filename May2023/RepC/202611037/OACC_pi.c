#include <stdio.h>
#include <omp.h>
#define N 1000000000
int main(void) {
double pi = 0.0f; long i;
double t_inicial, t_final;
t_inicial = omp_get_wtime();
#pragma acc parallel loop reduction(+:pi)
for (i=0; i<N; i++){
double t=(double) ((i+0.5)/N);
pi += 4.0/(1.0+t*t);
}
t_final = omp_get_wtime();
printf("pi=%f\n",pi/N);
printf("Tempo de execução: %lf\n", t_final - t_inicial);   
return 0;
}
