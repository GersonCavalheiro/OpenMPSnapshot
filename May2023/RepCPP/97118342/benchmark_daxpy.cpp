
#include <stdlib.h>
#include "benchmark.h"

extern "C" {

void daxpy_core(int array_size, FLOAT *a, FLOAT *b, FLOAT *c)
{
const FLOAT scalar = 3.141592;
int i;

#pragma omp parallel for
for (i=0; i<array_size; i++){
a[i] = b[i] + scalar * c[i];
}
}

double benchmark_daxpy(int max_step, int size) {
FLOAT *a;
FLOAT *b;
FLOAT *c;
double start_time, stop_time;

a = (FLOAT *)malloc(size * sizeof(FLOAT));
b = (FLOAT *)malloc(size * sizeof(FLOAT));
c = (FLOAT *)malloc(size * sizeof(FLOAT));
init_array_rand(size, b, 1.0);
init_array_rand(size, c, 1.0);

start_time = getTime();
for (int i=0; i<max_step; i++){
daxpy_core(size, a, b, c);
}
stop_time = getTime();

free(a);
free(b);
free(c);
return(stop_time - start_time);
}

}