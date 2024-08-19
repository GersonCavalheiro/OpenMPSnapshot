#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define WYMIAR 13
int main ()
{
double a[WYMIAR];
int i;
for(i = 0; i < WYMIAR; i++) a[i]=1.02*i;
double suma = 0.0;
for(i = 0; i < WYMIAR; i++) {
suma += a[i];
}
printf("Suma wyrazow tablicy: %lf\n", suma);
double suma_parallel = 0.0;
omp_set_num_threads(4);
#pragma omp parallel for ordered default(none) shared(suma_parallel, a) schedule(dynamic)
for(i = 0; i < WYMIAR; i++) {
int id_w = omp_get_thread_num();
#pragma omp critical(suma_parallel)
suma_parallel += a[i];
#pragma omp ordered
printf("a[%2d]->W_%1d  ", i, id_w); 
}
printf("\nSuma wyrazow tablicy rownolegle: %lf\n", suma_parallel);
return 0;
}