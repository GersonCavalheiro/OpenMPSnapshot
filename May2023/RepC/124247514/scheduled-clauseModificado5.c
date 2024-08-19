#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
main(int argc, char **argv) {
int i, n=200,chunk,a[n],suma=0, chunk_size, dyn, nthreads,intkind;
omp_sched_t kind;
omp_get_schedule(&kind, &chunk_size);
printf("Antes de modificar las variables...\n");
printf("dyn-var=%d\n",omp_get_dynamic());
printf("nthreads-var=%d\n",omp_get_max_threads());
printf("run-sched-var=%d\n",kind);
if(argc < 3) {
fprintf(stderr,"\nFormato: programa iteraciones chunk dyn-var nthreads run-sched-var \n");
exit(-1);
}
n = atoi(argv[1]); if (n>200) n=200; chunk = atoi(argv[2]); dyn = atoi(argv[3]); nthreads = atoi(argv[4]); intkind = atoi(argv[5]);
switch(intkind){
case 1: kind = omp_sched_static;
break;
case 2: kind = omp_sched_dynamic;
break;
case 3: kind = omp_sched_guided;
break;
case 4: kind = omp_sched_auto;
break;
}
omp_set_dynamic(dyn);
omp_set_num_threads(nthreads);
omp_set_schedule(kind,chunk);
omp_get_schedule(&kind, &chunk_size);
printf("\nDespués de modificar las variables...\n");
printf("dyn-var=%d\n",omp_get_dynamic());
printf("nthreads-var=%d\n",omp_get_max_threads());
printf("run-sched-var=%d\n",kind);
printf("\ndonde los valores de run-ched-var indican:\n1 → static\n2 → dynamic\n3 → guided\n4 → auto\n");
for (i=0; i<n; i++) a[i] = i;
#pragma omp parallel for firstprivate(suma) lastprivate(suma)
for (i=0; i<n; i++)
{
suma = suma + a[i];
printf(" thread %d suma a[%d]=%d suma=%d \n",
omp_get_thread_num(),i,a[i],suma);
}
printf("Fuera de 'parallel for' suma=%d\n",suma);
}
