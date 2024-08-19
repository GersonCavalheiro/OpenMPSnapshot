#pragma once
using namespace std; 
#include <iostream> 
#include <stdlib.h> 
#include <stdio.h> 
#include <omp.h> 


void showMatrix(const int* v, int n_elements) {
for (unsigned i = 0; i < n_elements; i++) {
printf(" % d ", v[i]);
}
printf("\n\n");
}

void initializeMatrixVector(int* v, int n_elements) {
for (unsigned i = 0; i < n_elements; i++) {
v[i] = rand() % 10;
}
}





static void matrixSumOMP1for(int* a, int* b, int* c, int n_elements, int nthreads) {


#pragma omp parallel for
for (int i = 0; i < n_elements; i++) {
c[i] = a[i] + b[i];
}
}


static void matrixSumOMP2forAndInternalLoopParallelized(int* a, int* b, int* c, int matrixWidth, int matrixHeight, int nthreads) {

#pragma omp parallel
{
for (int i = 0; i < matrixHeight; i++) {
#pragma omp for 
for (int j = 0; j < matrixWidth; j++) {
c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
}
}
}
}

static void matrixSumOMP2forAndExternalLoopParallelized(int* a, int* b, int* c, int matrixWidth, int matrixHeight, int nthreads) {

#pragma omp parallel
{
#pragma omp for 
for (int i = 0; i < matrixHeight; i++) {
for (int j = 0; j < matrixWidth; j++) {
c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
}
}
}
}

static void matrixSumOMP2forAndBothLoopsParallelized(int* a, int* b, int* c, const unsigned matrixWidth, const unsigned matrixHeight, const unsigned nthreads) {

omp_set_num_threads(nthreads);

#pragma omp parallel
{
#pragma omp for collapse(2)
for (int i = 0; i < matrixHeight; i++) {
for (int j = 0; j < matrixWidth; j++) {
c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
}
}
}
}


void matrixSumOpenMPOneLoop(int width, int height, int trials, int n_threads) {
printf("matrixSumOpenMP OneLoop");

const int WIDTH = width;
const int HEIGHT = height;
const int N_ELEMENTS = width * height;
size_t N_BYTES = N_ELEMENTS * sizeof(int);


int* a = (int*)malloc(N_BYTES * sizeof(int));
int* b = (int*)malloc(N_BYTES * sizeof(int));
int* c = (int*)malloc(N_BYTES * sizeof(int));

initializeMatrixVector(a, N_ELEMENTS);
initializeMatrixVector(b, N_ELEMENTS);

omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
matrixSumOMP1for(a, b, c, N_ELEMENTS, n_threads);

}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de matrixSumOMP1for() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH,HEIGHT, (t2 - t1) / (float)trials);


free(a);
free(b);
free(c);
}


void matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(int width, int height, int trials, int n_threads) {
printf("matrixSumOpenMP TwoLoops And InternalLoopParallelized");

const int WIDTH = width;
const int HEIGHT = height;
const int N_ELEMENTS = width * height;
size_t N_BYTES = N_ELEMENTS * sizeof(int);


int* a = (int*)malloc(N_BYTES);
int* b = (int*)malloc(N_BYTES);
int* c = (int*)malloc(N_BYTES);

initializeMatrixVector(a, N_ELEMENTS);
initializeMatrixVector(b, N_ELEMENTS);

omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
matrixSumOMP2forAndInternalLoopParallelized(a, b, c, WIDTH,HEIGHT, n_threads);

}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de matrixSumOMP2forAndInternalLoopParallelized() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH, HEIGHT, (t2 - t1) / (float)trials);


free(a);
free(b);
free(c);
}


void matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(int width, int height, int trials, int n_threads) {
printf("matrixSumOpenMP TwoLoops And ExternalLoopParallelized");

const int WIDTH = width;
const int HEIGHT = height;
const int N_ELEMENTS = width * height;
size_t N_BYTES = N_ELEMENTS * sizeof(int);


int* a = (int*)malloc(N_BYTES);
int* b = (int*)malloc(N_BYTES);
int* c = (int*)malloc(N_BYTES);

initializeMatrixVector(a, N_ELEMENTS);
initializeMatrixVector(b, N_ELEMENTS);

omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
matrixSumOMP2forAndExternalLoopParallelized(a, b, c, WIDTH, HEIGHT, n_threads);

}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de matrixSumOMP2forAndExternalLoopParallelized() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH, HEIGHT, (t2 - t1) / (float)trials);


free(a);
free(b);
free(c);
}


void matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(int width, int height, int trials, int n_threads) {
printf("matrixSumOpenMP TwoLoops And BothLoopsParallelized");

const int WIDTH = width;
const int HEIGHT = height;
const int N_ELEMENTS = width * height;
size_t N_BYTES = N_ELEMENTS * sizeof(int);


int* a = (int*)malloc(N_BYTES);
int* b = (int*)malloc(N_BYTES);
int* c = (int*)malloc(N_BYTES);

initializeMatrixVector(a, N_ELEMENTS);
initializeMatrixVector(b, N_ELEMENTS);

omp_set_num_threads(n_threads);
double t1, t2;
t1 = omp_get_wtime();
for (int i = 0; i < trials; i++)
{
matrixSumOMP2forAndBothLoopsParallelized(a, b, c, WIDTH, HEIGHT, n_threads);

}
t2 = omp_get_wtime();
printf("Tiempo medio en %d iteraciones de matrixSumOMP2forAndBothLoopsParallelized() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH, HEIGHT, (t2 - t1) / (float)trials);


free(a);
free(b);
free(c);
}

