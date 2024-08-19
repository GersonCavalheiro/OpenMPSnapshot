#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
main(int argc, char **argv) {
int i, n=200,chunk,a[n],suma=0, num_threads,num_proc,in_parallel;
if(argc < 3) {
fprintf(stderr,"\nFalta iteraciones o chunk \n");
exit(-1);
}
n = atoi(argv[1]); if (n>200) n=200; chunk = atoi(argv[2]);
for (i=0; i<n; i++) a[i] = i;
#pragma omp parallel for firstprivate(suma) lastprivate(suma,num_threads,num_proc,in_parallel)schedule(dynamic,chunk)
for (i=0; i<n; i++)
{ 
suma = suma + a[i];
printf(" thread %d suma a[%d]=%d suma=%d \n",
omp_get_thread_num(),i,a[i],suma);
num_threads = omp_get_num_threads();
num_proc = omp_get_num_procs();
in_parallel = omp_in_parallel();
}
printf("Dentro de 'parallel for' num_threads=%d\n",num_threads);
printf("Dentro de 'parallel for' num_procs=%d\n",num_proc);
printf("Dentro de 'parallel for' in_parallel=%d\n",in_parallel);
printf("Fuera de 'parallel for' suma=%d\n",suma);
printf("Fuera de 'parallel for' num_threads=%d\n",omp_get_num_threads());
printf("Fuera de 'parallel for' num_procs=%d\n",omp_get_num_procs());
printf("Fuera de 'parallel for' in_parallel=%d\n",omp_in_parallel());
}
