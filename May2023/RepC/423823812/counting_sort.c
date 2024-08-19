#include <stdlib.h>
#include "counting_sort.h"
#include "util.h"
#define MINMAX_1
#define COUNT_OCCURRANCE_1
#define POPULATE_1
void counting_sort(ELEMENT_TYPE *A, size_t A_len) 
{
if (A_len < 2)
{
return;
}
ELEMENT_TYPE min, max;
size_t *C;
size_t C_len, k;
min = A[0];
max = A[0];
for (size_t i = 1; i < A_len; i++)
{
if (A[i] < min)
{
min = A[i];
}
else if (A[i] > max)
{
max = A[i];
}
}
C_len = max - min + 1;
C = (size_t *) calloc(C_len, sizeof(size_t));
for (size_t i = 0; i < A_len; i++)
{
C[A[i] - min]++;
}
k = 0;
for (size_t i = 0; i < C_len; i++)
{
for (size_t j = 0; j < C[i]; j++)
{
A[k++] = i + min;
}
}
free(C);    
}
void counting_sort_parall1(ELEMENT_TYPE *A, size_t A_len, int threads) 
{
if (A_len < 2)
{
return;
}
ELEMENT_TYPE min, max;
size_t *C;
size_t C_len;
#ifdef MINMAX_1
min = A[0];
max = A[0];
#pragma omp parallel default(none) firstprivate(A_len, A) shared(max, min) num_threads(threads) 
{
ELEMENT_TYPE l_min = A[0];
ELEMENT_TYPE l_max = A[0];
#pragma omp for nowait
for (size_t i = 1; i < A_len; i++)
{
if (A[i] < l_min)
{
l_min = A[i];
}
else if (A[i] > l_max)
{
l_max = A[i];
}
}
#pragma omp critical
{
if (l_min < min)
{
min = l_min;
}
if (l_max > max)
{
max = l_max;
}
}
}
#endif
C_len = max - min + 1;
C = (size_t *) calloc(C_len, sizeof(size_t));
#if defined(COUNT_OCCURRANCE_1)
#pragma omp parallel default(none) firstprivate(A, A_len, C_len, min) shared(C) num_threads(threads)
{
size_t *C_loc = (size_t *) calloc(C_len, sizeof(size_t));
#pragma omp for nowait
for (size_t j = 0; j < A_len; j++)
{
C_loc[A[j] - min]++;
}
#pragma omp critical
{
for (size_t k = 0; k < C_len; k++)
{
C[k] += C_loc[k];
}
}
free(C_loc);
}
#elif defined(COUNT_OCCURRANCE_2)
#pragma omp parallel for default(none) shared(C) firstprivate(A, A_len, C_len, min) num_threads(threads)
for (size_t i = 0; i < A_len; i++)
{   
#pragma omp atomic
C[A[i] - min]++;
}
#endif
#if defined(POPULATE_1)  
for (size_t i = 1; i < C_len; i++)
{
C[i] += C[i-1];
}
#pragma omp parallel for default(none) firstprivate(A, C, C_len, min) num_threads(threads)
for (size_t i = 0; i < C_len; i++)
{
size_t start = (i != 0 ? C[i-1] : 0);
for (size_t j = start; j < C[i]; j++)
{
A[j] = i + min;
}
}
#elif defined(POPULATE_2)  
#pragma omp parallel for ordered default(none) firstprivate(C_len) shared(C) num_threads(threads)
for (size_t i = 1; i < C_len; i++)
{
#pragma omp ordered
{
C[i] += C[i-1];
}
}
#pragma omp parallel for default(none) firstprivate(A, C, C_len, min) num_threads(threads)
for (size_t i = 0; i < C_len; i++)
{
size_t start = (i != 0 ? C[i-1] : 0);
for (size_t j = start; j < C[i]; j++)
{
A[j] = i + min;
}
}
#endif
free(C); 
}
