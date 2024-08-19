#include <stdio.h>
#include <stdlib.h>
#define TRUE  1
#define FALSE 0
#include <omp.h>
int main(int argc, char *argv[]) {
int *a;
int n = 2; 
int nthreads, vlen, indx, offset = 4, i, TID;
int failed;
(void) omp_set_num_threads(3);
indx = offset;
#pragma omp parallel firstprivate(indx) shared(a,n,nthreads,failed)
{
#pragma omp single
nthreads = omp_get_num_threads();
vlen = indx + n*nthreads;
if ( (a = (int *) malloc(vlen*sizeof(int))) == NULL )
failed = TRUE;
else
failed = FALSE;
} 
for(i=0; i<vlen; i++) a[i] = -i-1;
printf("Comprimento do segmento por thread é %d\n", n);
printf("O offset do vetor a é %d\n",indx);
#pragma omp parallel default(none) firstprivate(indx) private(i,TID) shared(n,a)
{
TID = omp_get_thread_num();
indx += n*TID;
for(i=indx; i<indx+n; i++)
a[i] = TID + 1;
} 
printf("Depois da região paralela:\n");
for (i=0; i<vlen; i++)
printf("a[%d] = %d\n",i,a[i]);
free(a);
return(0);
}
