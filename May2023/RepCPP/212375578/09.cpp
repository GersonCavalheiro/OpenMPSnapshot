#include <cstdio>
#include <omp.h>

#include "lib.h"

int main() {
const int n = 6000;
const int m = 800;
int** d = random2dArray(n, m);

int min = d[0][0];
int max = d[0][0];

double t1 = omp_get_wtime(); 

#pragma omp parallel for
for (int i = 0; i < n; i++) {
int localMin = d[i][0];
int localMax = d[i][0];
for (int j = 0; j < m; j++) {
if (d[i][j] < localMin) {
localMin = d[i][j];
}
if (d[i][j] > localMax) {
localMax = d[i][j];
}
}
#pragma omp critical
if (localMin < min) {
min = localMin;
}
#pragma omp critical
if (localMax > max) {
max = localMax;
}
}

double t2 = omp_get_wtime(); 

printf("min=%d, max=%d\n", min, max);
printf("found in %f seconds\n", t2 - t1);
}
