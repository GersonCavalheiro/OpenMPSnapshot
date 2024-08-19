#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
irray = malloc(n_elements * sizeof *array);nt main(int nargs, char **args){
int N = 100;
double *a = malloc(N * sizeof *a);
double *b = malloc(N * sizeof *b);
double *c = malloc(N * sizeof *c);
for(int i=0; i<N; i++){
a[i] = i;
b[i] = i;
}
#pragma omp parallel num_threads(3) 
{
int blen, bstart, bend;
int thread_id, num_threads;
thread_id = omp_get_thread_num();
num_threads = omp_get_num_threads();
blen = N/num_threads;
if (thread_id < (N%num_threads)) {
blen = blen + 1;
bstart = blen * thread_id;
} else {
bstart = blen*thread_id + (N%num_threads);
}
bend = bstart + blen;
for(int i = bstart; i<bend; i++){
c[i] = a[i] + b[i];
}
}
for(int i=0; i<N; i++){
printf("%f\n", c[i]);
}
return 0;
}
