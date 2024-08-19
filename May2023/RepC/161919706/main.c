#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "multisort.h"
#include "merge.h"
int main()
{
int N;
int *A;
int *temp;
printf("Type how many numbers you want to sort: ");
scanf("%d", &N);
A=(int *)malloc(N*sizeof(int));
temp=(int *)malloc(N*sizeof(int));
for(int i=0;i<N;i++)
{
printf("A[%d]=", i);
scanf("%d", &A[i]);
}
printf("\"A\" array before multisort...\n\t");
for(int i=0;i<N;i++)
printf("%d ", A[i]);
printf("\n");
#pragma omp parallel shared(A)
#pragma omp single
multisort(&A[0], &temp[0], N);
printf("\"A\" array after multisort...\n\t");
for(int i=0;i<N;i++)
printf("%d ", A[i]);
printf("\n");
return 0;
}
