#include <cstdio>
#include <omp.h>

#include "lib.h"


int* multiply(int** matrix, int n, int m, int* vector) {
int* result = new int[n];
double t1 = omp_get_wtime();
#pragma omp parallel for
for (int i = 0; i < n; i++) {
int s = 0;
for (int j = 0; j < m; j++) {
s += matrix[i][j] * vector[j];
}
result[i] = s;
}
double t2 = omp_get_wtime();
printf("multiplied in %f s\n", (t2 - t1));
return result;
}

int main() {
int n = 20000;
int m = 20000;

int** matrix = random2dArray(n, m);
int*  vector = randomArray(m);

int* result = multiply(matrix, n, m, vector);

}

