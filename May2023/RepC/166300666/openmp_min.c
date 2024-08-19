#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "omp.h"
#define N 100
int main(void) {
srand(time(NULL));
double dizi[N] = {0.0}, minimum = 1.0;
int i;
for (i = 0; i < N; i++)
dizi[i] = (double)rand() / (double)RAND_MAX ;
#pragma omp parallel
{
#pragma omp for reduction(min:minimum) schedule(static)
for (i = 0; i < N; i++)
{
if ( dizi[i] < minimum )
minimum = dizi[i];
}
}
printf("Minimum = %f\n", minimum);
}
