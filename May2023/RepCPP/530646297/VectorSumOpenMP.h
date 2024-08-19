#pragma once
using namespace std; 
#include <iostream> 
#include <stdlib.h> 
#include <stdio.h> 
#include <omp.h> 


void showVector(const int* v, int n_elements) {
for (int i = 0; i < n_elements; i++) {
printf(" % d ", v[i]);
}
printf("\n\n");
}

void initializeVector(int* v, int n_elements) {
for (int i = 0; i < n_elements; i++) {
v[i] = rand() % 10;
}
}



static void vectorSumOMP(int* a, int* b, int* c, int n_elements) {


#pragma omp parallel for
for (int i = 0; i < n_elements; i++) {
c[i] = a[i] + b[i];
}
}

void vectorSumOpenMP(int n_elements, int trials, int n_threads) {

const int N_ELEMENTS = n_elements;
size_t N_BYTES = N_ELEMENTS * sizeof(int);

int* a = (int*)malloc(N_BYTES);
int* b = (int*)malloc(N_BYTES);
int* c = (int*)malloc(N_BYTES);

initializeVector(a, N_ELEMENTS);
initializeVector(b, N_ELEMENTS);

omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
vectorSumOMP(a, b, c, N_ELEMENTS);

}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de vectorSumOMP() con %d threads y con vector de %d elementos: % lf seconds.\n", trials, n_threads, N_ELEMENTS, (t2 - t1) / (float)trials);


free(a);
free(b);
free(c);
}
