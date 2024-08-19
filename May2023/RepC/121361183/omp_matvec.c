#include <omp.h>
#define SIZE 10
main ()
{
float A[SIZE][SIZE], b[SIZE], c[SIZE], total;
int i, j, tid;
total = 0.0;
for (i=0; i < SIZE; i++)
{
for (j=0; j < SIZE; j++)
A[i][j] = (j+1) * 1.0;
b[i] = 1.0 * (i+1);
c[i] = 0.0;
}
printf("\nStarting values of matrix A and vector b:\n");
for (i=0; i < SIZE; i++)
{
printf("  A[%d]= ",i);
for (j=0; j < SIZE; j++)
printf("%.1f ",A[i][j]);
printf("  b[%d]= %.1f\n",i,b[i]);
}
printf("\nResults by thread/row:\n");
#pragma omp parallel shared(A,b,c,total) private(tid,i)
{
tid = omp_get_thread_num();
#pragma omp for private(j)
for (i=0; i < SIZE; i++)
{
for (j=0; j < SIZE; j++)
c[i] += (A[i][j] * b[i]);
#pragma omp critical
{
total = total + c[i];
printf("  thread %d did row %d\t c[%d]=%.2f\t",tid,i,i,c[i]);
printf("Running total= %.2f\n",total);
}
}   
} 
printf("\nMatrix-vector total - sum of all c[] = %.2f\n\n",total);
}
