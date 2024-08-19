#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define N 1000000000

double f(double x) { return 4.0 / ( 1.0 + x * x); }

int main(void) {

double h = 1.0 / (double)N;
double pi = 0.0;
int i;

#pragma omp parallel for schedule(static) reduction(+:pi) private(i)
for (i = 0; i < N; i++) 
pi += f(i*h); 


pi *= h;

printf("pi = %.10f\n", pi);

}