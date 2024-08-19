#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define SIZE 1000
void swap(int *num1, int *num2)
{
int temp = *num1;
*num1 = *num2;
*num2 = temp;
}
int main(int argc, char *argv[])
{
int A[SIZE];
for (int i = 0; i < SIZE; i++)
{
A[i] = rand() % SIZE;
}
int N = SIZE;
int i = 0, j = 0;
int first;
double start, end;
start = omp_get_wtime();
for (i = 0; i < N - 1; i++)
{
first = i % 2;
#pragma omp parallel for num_threads(6) default(none), shared(A, first, N)
for (j = first; j < N - 1; j += 1)
{
if (A[j] > A[j + 1])
{
swap(&A[j], &A[j + 1]);
}
}
}
end = omp_get_wtime();
printf("time elapsed: %f\n\n",end-start);
for (i = 0; i < N; i++)
{
printf("%d ", A[i]);
}
printf("\n");
} 
