#include <stdio.h>
#include "omp.h"
#define N 1000
double f(double x) {
return 4.0 / ( 1.0 + x * x);
}
int main(void) {
double top = 0.0, x = 0.0;
int i;
double h = 1.0 / (double)N;
#pragma omp parallel private(i,x) shared(top)
{
#pragma omp for schedule(static) 
for (i = 0; i < N; i++) {
top = top + f(x);
x = i * h;      
}
top = top * h;
}
printf("PI = %f\n", top);
return 0;
}
