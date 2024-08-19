#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(void) {
int np = omp_get_max_threads();
if (np<2) exit(1);
int * A = malloc(np*sizeof(int));
#pragma omp parallel shared(A)
{
int me = omp_get_thread_num();
int B = 134;
if (me==0) A[1] = B;
#pragma omp barrier
if (me==1) printf("A@1=%d\n",A[1]); fflush(stdout);
}
free(A);
return 0;
}
