#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define SIZE 50000000
void calc_stuff(void);
int main() {
calc_stuff();
return 0;
}
void calc_stuff() {
double *c = (double *)malloc(sizeof(double) * SIZE);
#pragma omp parallel for
for (int i = 0; i < SIZE; i++) {
c[i] = sqrt(i * 32) + sqrt(i * 16 + i * 8) + sqrt(i * 4 + i * 2 + i);
c[i] -= sqrt(i * 32 * i * 16 + i * 4 + i * 2 + i);
c[i] += pow(i * 32, 8) + pow(i * 16, 12);
}
free(c);
}
