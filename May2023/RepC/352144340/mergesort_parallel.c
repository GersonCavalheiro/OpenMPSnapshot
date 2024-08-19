#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
void generate_list(int * x, int n) {
int i;
srand (time (NULL));
for (i = 0; i < n; i++)
x[i] = rand() % n; 
}
void print_list(int * x, int n) {
int i;
for (i = 0; i < n; i++) {
printf("%d ",x[i]);
} 
}
void merge(int * X, int n, int * tmp) {
int i = 0;
int j = n/2;
int ti = 0;
while (i<n/2 && j<n) { 
if (X[i] < X[j]) {
tmp[ti] = X[i];
ti++; i++;
} else {
tmp[ti] = X[j];
ti++; j++;
}
}
while (i<n/2) { 
tmp[ti] = X[i];
ti++; i++;
}
while (j<n) { 
tmp[ti] = X[j];
ti++; j++;
}
memcpy(X, tmp, n*sizeof(int));
} 
void mergesort(int * X, int n, int * tmp)
{
if (n < 2) return;
#pragma omp task
{
mergesort(X, n/2, tmp); 
}
#pragma omp task
{
mergesort(X+(n/2), n-(n/2), tmp);
}
#pragma omp taskwait
merge(X, n, tmp);
}
void main(int argc, char *argv[])
{
int n;
int *data, *tmp;
double start, end;
if (argc != 2) {
printf ("Usage : %s <list size>\n", argv[0]);
}
n = strtol(argv[1], NULL, 10);
data = (int *) malloc (sizeof(int)*n);
tmp = (int *) malloc (sizeof(int)*n);
generate_list(data, n);
printf("List Before Sorting...\n");
start = omp_get_wtime();
#pragma omp parallel
#pragma omp single
mergesort(data, n, tmp);
end = omp_get_wtime();
printf("\nList After Sorting...\n");
printf("Time: %f\n",(end-start));
printf("\n");
}
