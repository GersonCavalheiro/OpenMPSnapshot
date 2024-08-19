#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
void Mat_Init(int row, int col, double *X) {
int i, size;
size = row * col;
for (i = 0; i < size; i++) {
X[i] = (double) (rand() % 10 + 1);
}
}
void Mat_Show_p(int row, int col, double *X, int precision) {
int i, j;
printf("row = %d col = %d\n", row, col);
for (i = 0; i < row; i++) {
for (j = 0; j < col; j++) {
if (precision < 0) {
printf("%lf ", X[col * i + j]);
} else {
printf("%.*f ", precision, X[col * i + j]);
}
}
printf("\n");
}
}
void Mat_Show(int row, int col, double *X) {
Mat_Show_p(row, col, X, -1);
}
void Mat_Clone(int row, int col, double *Src, double *Dest) {
int i, size;
size = row * col;
for (i = 0; i < size; i++) {
Dest[i] = Src[i];
}
}
void Mat_Xv(int row, int col, double *X, double *Y, double *v) {
int i, j;
double result;
for (i = 0; i < row; i++) {
result = 0;
for (j = 0; j < col; j++)
result += v[j] * X[col * i + j];
Y[i] = result;
}
}
void Omp_Mat_Xv(int row, int col, double *X, double *Y, double *v, int thread_count) {
int i, j;
double result;
#pragma omp parallel for num_threads(thread_count) default(none) private(i, j, result) shared(row, col, X, Y, v)
for (i = 0; i < row; i++) {
result = 0;
for (j = 0; j < col; j++)
result += v[j] * X[col * i + j];
Y[i] = result;
}
}