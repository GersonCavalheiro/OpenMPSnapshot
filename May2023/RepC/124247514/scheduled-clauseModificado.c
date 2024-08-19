#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
main(int argc, char **argv) {
int i, n=200,chunk,a[n],suma=0, modifier, dyn, max, limit;
omp_sched_t kind;
if(argc < 3) {
fprintf(stderr,"\nFalta iteraciones o chunk \n");
exit(-1);
}
n = atoi(argv[1]); if (n>200) n=200; chunk = atoi(argv[2]);
for (i=0; i<n; i++) a[i] = i;
#pragma omp parallel for firstprivate(suma) lastprivate(suma,dyn,max,limit)schedule(dynamic,chunk)
for (i=0; i<n; i++)
{ 
suma = suma + a[i];
printf(" thread %d suma a[%d]=%d suma=%d \n",
omp_get_thread_num(),i,a[i],suma);
dyn = omp_get_dynamic();
max = omp_get_max_threads();
limit = omp_get_thread_limit();
omp_get_schedule(&kind, &modifier);
}
printf("Dentro de 'parallel for' dyn-var=%d\n",dyn);
printf("Dentro de 'parallel for' nthreads-var=%d\n",max);
printf("Dentro de 'parallel for' thread-limit-var=%d\n",limit);
printf("Dentro de 'parallel for' run-sched-var=%d\n",kind);
omp_get_schedule(&kind, &modifier);
printf("Fuera de 'parallel for' suma=%d\n",suma);
printf("Fuera de 'parallel for' dyn-var=%d\n",omp_get_dynamic());
printf("Fuera de 'parallel for' nthreads-var=%d\n",omp_get_max_threads());
printf("Fuera de 'parallel for' thread-limit-var=%d\n",omp_get_thread_limit());
printf("Fuera de 'parallel for' run-sched-var=%d\n",kind);
}
