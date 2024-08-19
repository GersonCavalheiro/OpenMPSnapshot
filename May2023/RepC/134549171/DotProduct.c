#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
#define N 1000000
double A[N], B[N];
int main(){
double sum;
double start_time, run_time;
int n=N;
int i;
for (i = 0; i < n; i++) {
A[i] = i * 0.5;
B[i] = i * 2.0;
}
sum = 0;
start_time = omp_get_wtime();
#pragma omp parallel for reduction(+:sum)
for (i = 0; i < n; i++ ) {
sum = sum + A[i]*B[i];
}
printf ("sum = %f \n", sum);
run_time = omp_get_wtime() - start_time;
printf("%f\n", run_time);
return 0;
}
