

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "sort.hh"


void partition (keytype pivot, int N, keytype* A,
int* p_n_lt, int* p_n_eq, int* p_n_gt)
{

int n_lt = 0, n_eq = 0, n_gt = 0;
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
if (A[i] < pivot) ++n_lt;
else if (A[i] == pivot) ++n_eq;
else ++n_gt;
}

keytype* A_orig = newCopy (N, A);


int i_lt = 0; 
int i_eq = n_lt; 
int i_gt = n_lt + n_eq; 
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
keytype ai = A_orig[i];
if (ai < pivot)
A[i_lt++] = ai;
else if (ai > pivot)
A[i_gt++] = ai;
else
A[i_eq++] = ai;
}
assert (i_lt == n_lt);
assert (i_eq == (n_lt+n_eq));
assert (i_gt == N);

free (A_orig);

if (p_n_lt) *p_n_lt = n_lt;
if (p_n_eq) *p_n_eq = n_eq;
if (p_n_gt) *p_n_gt = n_gt;
}

void
quickSort (int N, keytype* A)
{
const int G = 1250000; 
if (N < G)
sequentialSort (N, A);
else {
keytype pivot = A[rand () % N];

int n_less = -1, n_equal = -1, n_greater = -1;
partition (pivot, N, A, &n_less, &n_equal, &n_greater);
assert (n_less >= 0 && n_equal >= 0 && n_greater >= 0);
#pragma omp task
quickSort (n_less, A);
quickSort (n_greater, A + n_less + n_equal);
#pragma omp taskwait
}
}

void
parallelSort (int N, keytype* A)
{
#pragma omp parallel
#pragma omp single
quickSort (N, A);
}


