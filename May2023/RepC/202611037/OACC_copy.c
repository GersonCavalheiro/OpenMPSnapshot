#include <stdlib.h>
#include <omp.h>
int main(int argc, char **argv){
int n = 10000; 
double t_inicial, t_final;
if (argc > 1)
n = atoi(argv[1]);
double a[n];
t_inicial = omp_get_wtime();
#pragma acc data copy(a)
{
#pragma acc parallel loop
for(int i = 0; i < n; i++){
a[i] = 0.0;
}
}
t_final = omp_get_wtime();
printf("tempo: %3.2f\n", t_final - t_inicial);
return 0;
}
