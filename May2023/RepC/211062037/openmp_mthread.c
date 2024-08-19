#include <time.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
void print_matrix(int **mat, int m, int n);
void bubblesort(int *mat, int n);
void quicksort(int *mat, int n);
void swap(int *x, int *y);
int cmpfunc (const void * a, const void * b) {
return ( *(int*)a - *(int*)b );
}
int **global_var_mat;
void quicksort(int *mat, int n)
{
if (n < 2)
return;
int pivot = mat[n / 2];
int i, j;
for (i = 0, j = n - 1;; i++, j--)
{
while (mat[i] < pivot)
i++;
while (mat[j] > pivot)
j--;
if (i >= j)
break;
int temp = mat[i];
mat[i] = mat[j];
mat[j] = temp;
}
quicksort(mat, i);
quicksort(mat + i, n - i);
}
void bubblesort(int *mat, int n)
{
for (int step = 0; step < n - 1; ++step)
{
for (int i = 0; i < n - step - 1; ++i)
{
if (mat[i] > mat[i + 1])
{
swap(&mat[i], &mat[i + 1]);
}
}
}
}
void swap(int *x, int *y)
{
int t;
t = *x;
*x = *y;
*y = t;
}
void print_matrix(int **mat, int m, int n)
{
for (int i = 0; i < m; ++i)
{
for (int j = 0; j < n; ++j)
{
printf("%d ", mat[i][j]);
}
printf("\n");
}
printf("\n");
}
int main(int argc, char const *argv[])
{
int m = 40, n = 40;
global_var_mat = (int **)malloc(m * sizeof(int *));
for (int i = 0; i < m; i++)
{
global_var_mat[i] = (int *)malloc(n * sizeof(int));
}
for (int i = 0; i < m; ++i)
for (int j = 0; j < n; ++j)
global_var_mat[i][j] = rand() % 100 + 1;
print_matrix(global_var_mat, m, n);
#pragma omp parallel
for (size_t tnum = 0; tnum < m; tnum++)
{
quicksort(global_var_mat[tnum], n);
}
print_matrix(global_var_mat, m, n);
for (int i = 0; i < m; ++i)
free(global_var_mat[i]);
free(global_var_mat);
exit(EXIT_SUCCESS);
}
