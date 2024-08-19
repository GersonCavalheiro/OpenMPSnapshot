#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#define N 100000
double f(double x) { 
return 4.0 / ( 1.0 + x * x );
}
int main(void) {
double h = 1.0 / (double)N;
double x, sum, sonuc;
int i, tid, count;
#pragma omp parallel private(i, x) shared(sonuc, sum)
{
#pragma omp for reduction(+:sum) schedule(runtime)
for (i = 0; i < N; i++) {
x = (double)i * h;     
sum += f( x );
}
#pragma omp master
sum = sum * h;
}
printf("Sum = %f\n", sum);
}
