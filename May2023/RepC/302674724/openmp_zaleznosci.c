#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
#include<math.h>
#define N 1000000
int main(){
int i;
double* A = malloc((N+2)*sizeof(double));
double* B = malloc((N+2)*sizeof(double));
double* C = malloc((N+2)*sizeof(double));
double suma;
for(i=0;i<N+2;i++) A[i] = (double)i/N;
for(i=0;i<N+2;i++) B[i] = 1.0 - (double)i/N;
double t1 = omp_get_wtime();
for(i=0; i<N; i++){
A[i] += A[i+2] + sin(B[i]);
}
t1 = omp_get_wtime() - t1;
suma = 0.0;
for(i=0;i<N+2;i++) suma+=A[i];
printf("suma %lf, czas obliczen %lf\n", suma, t1);
for(i=0;i<N+2;i++) A[i] = (double)i/N;
for(i=0;i<N+2;i++) B[i] = 1.0 - (double)i/N;
t1 = omp_get_wtime();
omp_set_num_threads(10);
#pragma omp parallel for
for(i = 0; i < N; i++){
C[i] += A[i] + A[i+2] + sin(B[i]);
}
t1 = omp_get_wtime() - t1;
suma = 0.0;
#pragma omp parallel for reduction(+ : suma)
for(i = 0; i < N+2; i++) suma += C[i];
printf("suma %lf, czas obliczen rownoleglych %lf\n", suma, t1);
}
