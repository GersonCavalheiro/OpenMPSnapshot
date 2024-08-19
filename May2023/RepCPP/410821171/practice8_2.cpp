#include <iostream>
#include <omp.h>
#include <algorithm>

#define ROWS 4
#define COLS 1000000

void mxv(float(*matrix_d)[COLS], float* vector, float* result) {
#pragma omp parallel for num_threads(4)
for (int y = 0; y < ROWS; y++) {
for (int x = 0; x < COLS; x++) {
result[y] += matrix_d[y][x] * vector[x];
}
}
}

void mxv_local(float (*matrix_d)[COLS], float* vector, float* result) {
float local_result = 0.0;
#pragma omp parallel for num_threads(4) private(local_result)
for (int y = 0; y < ROWS; y++) {
local_result = 0.0;
for (int x = 0; x < COLS; x++) {
local_result += matrix_d[y][x] * vector[x];
}
result[y] += local_result;
}
}

void mxv_dummy(float(*matrix_d)[COLS], float* vector, float(*result)[COLS]) {
#pragma omp parallel for num_threads(4)
for (int y = 0; y < ROWS; y++) {
for (int x = 0; x < COLS; x++) {
result[y][0] += matrix_d[y][x] * vector[x];
}
}
}

int main() {
float matrix[ROWS][COLS], results[ROWS], vec[COLS], results_dummy[ROWS][COLS];
int i, j = 0;
double start_time, end_time;

for (j = 0; j < ROWS; j++) {
for (i = 0; i < COLS; i++) {
matrix[j][i] = (((float)(j+1)) * COLS * (float)i) / ((float)1000.);
}
}

std::fill(results, results + ROWS, 0.0);

for (i = 0; i < COLS; i++) vec[i] = ((float)i) / (float)1000.;


#pragma omp barrier
start_time = omp_get_wtime();
mxv_dummy(matrix, vec, results_dummy);
end_time = omp_get_wtime();

for (i = 0; i < ROWS; i++) std::cout << "resi=" << results_dummy[i][0] << "\n";

std::cout << "\n Exe. Time = " << (end_time - start_time) * 1000. << " msec\n";
return 0;
}