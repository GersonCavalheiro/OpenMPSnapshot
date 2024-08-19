#include <iostream>
#include <cmath>
#include <omp.h>

#define MAX_ITERATIONS 100000
#define SIZE 45000

#define E 0.000000000001


double * getArray(int size){
double * arr = (double *) calloc(size, sizeof(double));
return arr;
}

void vector_sub(double *vec1, double *vec2, double *result) {
#pragma omp parallel for
for (int i = 0; i < SIZE; i++) {
result[i] = vec1[i] - vec2[i];
}
}

void matrix_vector_mul(double *matrix, double *vector, double *result) {
#pragma omp parallel for
for (int i = 0; i < SIZE; i++){
result[i]=0;
for (int j = 0; j < SIZE; j++){
result[i] += matrix[i * SIZE + j] * vector[j];
}
}
}


double scalar(double *v1, double *v2) {
double result = 0;
#pragma omp parallel for reduction(+: result)
for (int i = 0; i < SIZE; i++) {
result += v1[i] * v2[i];
}
return result;
}

double norm(double *vector) {
double result = 0;
#pragma omp parallel for reduction(+: result)
for (int i = 0; i < SIZE; i++) {
result += vector[i] * vector[i];
}
return sqrt(result);
}


void fill(double *matrix, double *b, double *x) {
#pragma omp parallel for
for (int i = 0; i < SIZE; i++) {
for (int j = 0; j < SIZE; j++) {
if (i == j) {
matrix[i * SIZE + j] = 2.0;
} else {
matrix[i * SIZE + j] = 1.0;
}
}
b[i] = SIZE + 1;
x[i] = 0;
}
}


void nextX(double *x, double tetta, double *y) {
#pragma omp parallel for
for (int i = 0; i < SIZE; ++i) {
x[i] -= tetta * y[i];
}
}

void nextY(double *y, double * A, double *x, double *b){
double * tmp = getArray(SIZE);
matrix_vector_mul(A, x, tmp);
vector_sub(tmp, b, y);

}

double getTetta(double * A, double * y){
double * tmp = getArray(SIZE);
matrix_vector_mul(A, y, tmp);
double tetta = scalar(y, tmp) / scalar(tmp, tmp);
free(tmp);
return tetta;
}


void print_vector(double * vec){
for (int i = 0; i < SIZE; ++i) {
printf("%f ",vec[i]);
}
printf("\n");
}

int main(int argc, char *argv[]) {

double *A = getArray(SIZE * SIZE);
double *b = getArray(SIZE);
double *x = getArray(SIZE);
double *y = getArray(SIZE);
double start = omp_get_wtime();
omp_set_num_threads(8);
fill(A,b,x);
double tetta = 0;
double normB = norm(b);
nextY(y, A, x, b);
int iterations = 0;
while (norm(y) / normB > E && iterations++ < MAX_ITERATIONS) {
tetta = getTetta(A, y);
nextX(x, tetta, y);
nextY(y, A, x, b);
}
std::cout << "Time = " << omp_get_wtime() - start << std::endl;
free(A);
free(b);
free(x);
free(y);
return 0;
}