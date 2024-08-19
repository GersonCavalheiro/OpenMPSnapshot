#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#define N 1000
double f(double x) { 
return 4.0 / ( 1.0 + x * x );
}
int main(void) {
double h = 1.0 / (double)N;
double x, sum, sonuc;
int i, tid, count;
#pragma omp parallel private(i, x, count, sum, tid) shared(sonuc)
{
tid = omp_get_thread_num();
count = 0;
#pragma omp for schedule(runtime)
for (i = 0; i < N; i++) {
x = (double)i * h;     
sum += f( x );
count++;
}
printf("Thread %d --> count = %d\n", tid, count);
#pragma omp critical
sonuc += sum;
}
sonuc *= h;
printf("PI = %f\n", sonuc);
}
