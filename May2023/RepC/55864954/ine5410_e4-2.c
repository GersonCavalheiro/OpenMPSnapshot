#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define NRA 800 
#define NCA 800 
#define NCB 800 
void matrix_mult(void);
int main() {
matrix_mult();
return 0;
}
void matrix_mult() {
int i, j, k;
double **a = (double **)malloc(sizeof(double *) * NRA);
for (i = 0; i < NRA; i++) {
a[i] = (double *)malloc(sizeof(double) * NCA);
}
double **b = (double **)malloc(sizeof(double *) * NCA);
for (i = 0; i < NCA; i++) {
b[i] = (double *)malloc(sizeof(double) * NCB);
}
double **c = (double **)malloc(sizeof(double *) * NRA);
for (i = 0; i < NRA; i++) {
c[i] = (double *)malloc(sizeof(double) * NCB);
}
for (i = 0; i < NRA; i++) {
for (j = 0; j < NCA; j++) {
a[i][j] = i + j;
}
}
for (i = 0; i < NCA; i++) {
for (j = 0; j < NCB; j++) {
b[i][j] = i * j;
}
}
for (i = 0; i < NRA; i++) {
for (j = 0; j < NCB; j++) {
c[i][j] = 0;
}
}
#pragma omp parallel for private(i, j, k)
for (i = 0; i < NRA; i++) {
for (j = 0; j < NCB; j++) {
for (k = 0; k < NCA; k++) {
c[i][j] += a[i][k] * b[k][j];
}
}
}
for (i = 0; i < NRA; i++) {
for (j = 0; j < NCB; j++) {
printf("%10.2f  ", c[i][j]);
}
printf("\n");
}
for (i = 0; i < NRA; i++) {
free(a[i]);
}
free(a);
for (i = 0; i < NCA; i++) {
free(b[i]);
}
free(b);
for (i = 0; i < NRA; i++) {
free(c[i]);
}
free(c);
}
