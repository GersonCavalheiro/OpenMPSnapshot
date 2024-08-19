#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
int main(void) {
int np = omp_get_max_threads();
if (np<2) exit(1);
int ** A = malloc(np*sizeof(int*));
#pragma omp parallel shared(A)
{
int me = omp_get_thread_num();
A[me] = malloc(sizeof(int));
#pragma omp barrier
int B = 0x86;
if (me==0) A[1][0] = B;
#pragma omp barrier
if (me==1) printf("A@1=%d\n",A[1][0]);
free(A[me]);
}
free(A);
return 0;
}
