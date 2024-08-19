#include <stdio.h>
#include <stdlib.h>
#define TRUE  1
#define FALSE 0
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
int main()
{
int *a;
int n = 2, nthreads, vlen, indx, offset = 4, i, TID;
int failed;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
indx = offset;
#pragma omp parallel firstprivate(indx) shared(a,n,nthreads,failed)
{
#pragma omp single
{
nthreads = omp_get_num_threads();
vlen = indx + n*nthreads;
if ( (a = (int *) malloc(vlen*sizeof(int))) == NULL )
failed = TRUE;
else
failed = FALSE;
}
} 
if ( failed == TRUE ) {
printf("Fatal error: memory allocation for a failed vlen = %d\n",vlen);
return(-1);
}
else
{
printf("Diagnostics:\n");
printf("nthreads = %d\n",nthreads);
printf("indx     = %d\n",indx);
printf("n        = %d\n",n);
printf("vlen     = %d\n",vlen);
}
for(i=0; i<vlen; i++) a[i] = -i-1;
printf("Length of segment per thread is %d\n",n);
printf("Offset for vector a is %d\n",indx);
#pragma omp parallel default(none) firstprivate(indx) private(i,TID) shared(n,a)
{
TID = omp_get_thread_num();
indx += n*TID;
for(i=indx; i<indx+n; i++)
a[i] = TID + 1;
} 
printf("After the parallel region:\n");
for (i=0; i<vlen; i++)
printf("a[%d] = %d\n",i,a[i]);
free(a);
return(0);
}
