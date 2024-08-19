#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
int calculate_sum(int length);
int *pglobal;
#pragma omp threadprivate(pglobal)
int main()
{
int i, j, sum, TID, n = 5;
int length[n], check[n];
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
for (i=0; i<n; i++)
{
length[i] = 10 * (i+1);
check[i]  = length[i]*(length[i]+1)/2; 
}
#pragma omp parallel for shared(n,length,check) private(TID,i,j,sum)
for (i=0; i<n; i++)
{
TID = omp_get_thread_num();
if ( (pglobal = (int *) malloc(length[i]*sizeof(int))) != NULL ) {
for (j=sum=0; j<length[i]; j++)
pglobal[j] = j+1;
sum = calculate_sum(length[i]);
printf("TID %d: value of sum for i = %d is %8d (check = %8d)\n",
TID,i,sum,check[i]);
free(pglobal);
} else {
printf("TID %d: fatal error in malloc for length[%d] = %d\n",
TID,i,length[i]);
}
} 
return(0);
}
int calculate_sum(int length)
{
int sum = 0;
for (int j=0; j<length; j++)
sum += pglobal[j];
return(sum);
}
