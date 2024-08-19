#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ompvv.h"
#define rowA 500        
#define colA 500        
#define colB 500        
int main (int argc, char *argv[]) 
{
OMPVV_TEST_OFFLOADING;
int tid, nthreads, i, j, k;
int	*a = (int*) malloc(sizeof(int) * rowA * colA);           
int	*b = (int*) malloc(sizeof(int) * colA * colB);           
int	*c = (int*) malloc(sizeof(int) * rowA * colB);           
for (i = 0; i < rowA; i++)
for (j = 0; j < colA; j++)
a[i*rowA+j] = 10; 
for (i = 0; i < colA; i++)
for (j = 0; j < colB; j++)
b[i*colA+j] = 50; 
for (i = 0; i < rowA; i++)
for (j = 0; j < colB; j++)
c[i*rowA+j] = 0;
int DimA = rowA*colA;
int DimB = colB*colA;
int DimC = rowA*colA;
#pragma omp target map(to: a[0:DimA], b[0:DimB]) map(from: c[0:DimC])
{
#pragma omp teams distribute parallel for simd collapse(2) private(k)
for (i = 0; i < rowA; i++)
for(j = 0; j < colB; j++)
for(k = 0; k < colA; k++)
c[i*rowA+j] = a[i*rowA+j] * b[k*colA+j];
}
int error = 0;
for (i = 0; i < rowA; i++)
{
for (j = 0; j < colB; j++) {
OMPVV_TEST_AND_SET(error, 500 != c[i*rowA+j]);
OMPVV_ERROR_IF(500 != c[i*rowA+j], "Error: [%d][%d] should be 500 is %d",i,j,c[i*rowA+j]);
}
}
free(a);
free(b);
free(c);
OMPVV_REPORT_AND_RETURN(error);
}
