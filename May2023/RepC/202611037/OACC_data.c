#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int main(int argc, char **argv){
float A[1000];
double t_inicial, t_final;
t_inicial = omp_get_wtime();
#pragma acc data copyin(A)
{
#pragma acc kernels
{
for(int i = 1; i < 1000; i++){
A[i] = 1.0;
}
}
A[10] = 2.0;
}
t_final = omp_get_wtime();
printf("tempo: %3.2f\n", t_final - t_inicial);
printf("A[10] = %f\n", A[10]);
return 0;
}
