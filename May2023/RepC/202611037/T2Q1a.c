#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define TAM 50000
int main (int argc, char *argv[]) {
int k, n = 40000, x[TAM], y[TAM], z[TAM], q=10, r=2, t=2;
double t_inicial, t_final;
for(k=0; k<TAM; k++){
y[k]=k;
z[k]=k;
}
t_inicial = omp_get_wtime();
#pragma omp simd
for( k=0 ; k<n ; k++ ) {
x[k] = q + y[k]*( r*z[k+10] + t*z[k+11] );
}
t_final = omp_get_wtime();
printf("tempo: %lf", t_final-t_inicial);
return 0;
}