#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#endif
int main()
{
int n = 4;
int *a, **b;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
if ( (a=(int *)malloc(n*sizeof(int))) == NULL ) {
perror("array a"); exit(-1);
}
if ( (b=(int **)malloc(n*sizeof(int *))) == NULL ) {
perror("array b"); exit(-1);
} 
else {
for (int i=0; i<n; i++)
if ( (b[i]=(int *)malloc(n*sizeof(int))) == NULL )
{perror("array b"); exit(-1);}
}
#pragma omp parallel shared(n,a,b)
{
#pragma omp for
for (int i=0; i<n; i++)
{
a[i] = i + 1;
#pragma omp parallel for 
for (int j=0; j<n; j++)
b[i][j] = a[i];
}
} 
for (int i=0; i<n; i++)
{
for (int j=0; j<n; j++)
printf("b[%d][%d] = %d ",i,j,b[i][j]);
printf("\n");
}
free(a);
free(b);
}
