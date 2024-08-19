#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "omp.h"
#include <mpi.h>
#define MAX_SIZE 1000
void generate_list(int * x, int n) 
{
int i,j,t;
for (i = 0; i < n; i++)
x[i] = i;
for (i = 0; i < n; i++) 
{
j = rand() %n;
t = x[i];
x[i] = x[j];
x[j] = t;
}
}
void print_list(int * x, int n) 
{
int i;
for (i = 0; i < n; i++) 
{
printf("%d ",x[i]);
}
}
void merge(int * X, int n, int * tmp) 
{
int i = 0;
int j = n/2;
int ti = 0;
while (i<n/2 && j<n) 
{
if (X[i] < X[j]) 
{
tmp[ti] = X[i];
ti++;
i++;
}
else 
{
tmp[ti] = X[j];
ti++; 
j++;
}
}
while (i<n/2) 
{
tmp[ti] = X[i];
ti++; 
i++;
}
while (j<n)
{ 
tmp[ti] = X[j];
ti++;
j++;
}
memcpy(X, tmp, n*sizeof(int));
}
void mergesort(int * X, int n, int * tmp)
{
if (n < 2) 
return;
#pragma omp task firstprivate (X, n, tmp)
mergesort(X, n/2, tmp);
#pragma omp task firstprivate (X, n, tmp)
mergesort(X+(n/2), n-(n/2), tmp);
#pragma omp taskwaitmerge(X, n, tmp);
}
int main(int argc, char *argv[])
{
int n;
double start, stop;
int data[MAX_SIZE], tmp[MAX_SIZE];
int rank,numtasks;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
printf("enter the number of elements to sort:\n");
scanf("%d",&n);
generate_list(data, n);
printf("\nList Before Sorting...\n");
print_list(data, n);
start = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
mergesort(data, n, tmp);
}
stop = omp_get_wtime();
printf("\n\nList After Sorting...\n");
print_list(data, n);
printf("\n\nTime: %g\n",stop-start);
MPI_Finalize();
return 0;
}   
