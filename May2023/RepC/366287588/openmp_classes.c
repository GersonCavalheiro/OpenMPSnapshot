#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>

#define MAX_THREAD_COUNT 512

int **a, **b;
int dim;

void getMatrixFromFile(int n){



a = (int **)malloc(n * sizeof(int*));

for(int i = 0; i<n; i++){
a[i] = (int *)malloc(n * sizeof(int));
}

b = (int **)malloc(n * sizeof(int*));

for(int i = 0; i<n; i++){
b[i] = (int*)malloc(n * sizeof(int));
}



for(int i = 0; i<n; i++){
for(int j = 0; j<n; j++){
a[i][j] = rand() % 100;
}
}

for(int i = 0; i<n; i++){
for(int j = 0; j<n; j++){
b[i][j] = rand() % 100;
}
}
dim = n;
}


int main(int argc, char** argv){

int n = atoi(argv[1]);
getMatrixFromFile(n);

double start_static, start_dynamic, start_guided;
double time_static, time_dynamic, time_guided;
int c[dim][dim];
int no_of_threads = MAX_THREAD_COUNT > dim*dim ? dim*dim : MAX_THREAD_COUNT;

start_static = omp_get_wtime(); 

#pragma omp parallel default(none) shared (a, b, c, dim) num_threads(no_of_threads)
#pragma omp for collapse(2) schedule(static)
for (int i = 0; i < dim; i++) {
for (int j = 0; j < dim; j++) {
c[i][j] = 0;
for (int k = 0; k < dim; k++) {
c[i][j] += a[i][k] * b[k][j];
}

}
}
time_static = omp_get_wtime() - start_static;



start_dynamic = omp_get_wtime(); 

#pragma omp parallel default(none) shared (a, b, c, dim) num_threads(no_of_threads)
#pragma omp for collapse(2) schedule(dynamic)
for (int i = 0; i < dim; i++) {
for (int j = 0; j < dim; j++) {
c[i][j] = 0;
for (int k = 0; k < dim; k++) {
c[i][j] += a[i][k] * b[k][j];
}

}
}
time_dynamic = omp_get_wtime() - start_dynamic;



start_guided = omp_get_wtime();
#pragma omp parallel default(none) shared (a, b, c, dim) num_threads(no_of_threads)
#pragma omp for collapse(2) schedule(guided)
for (int i = 0; i < dim; i++) {
for (int j = 0; j < dim; j++) {
c[i][j] = 0;
for (int k = 0; k < dim; k++) {
c[i][j] += a[i][k] * b[k][j];
}

}
}
time_guided = omp_get_wtime() - start_guided;




printf("*******************************************\n");

printf("Time Taken in static scheduling: %lf\n", (time_static));
printf("Time Taken in dynamic scheduling: %lf\n", (time_dynamic));
printf("Time Taken in guided scheduling: %lf\n", (time_guided));

}
