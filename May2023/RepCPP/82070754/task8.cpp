#include "iostream"
#include "omp.h"
#include "../../array_utils.h"

const int SIZE = 16000;

int main() {

int *a = new int[SIZE];
for (int i = 0; i < SIZE; i++) {
a[i] = i;
}

double *b = new double[SIZE];
b[0] = (double) a[0];
b[SIZE - 1] = (double) a[SIZE - 1];

double start_time = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(8)
for (int i = 1; i < SIZE - 1; i++) {
b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
}
double end_time = omp_get_wtime();
printf("[STATIC] : %2.15f\n", end_time - start_time);

start_time = omp_get_wtime();
int chunk = 200;
#pragma omp parallel for schedule(dynamic, chunk) num_threads(8)
for (int i = 1; i < SIZE - 1; i++) {
b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
}
end_time = omp_get_wtime();
printf("[DYNAMIC WITH %d CHUNK] : %2.15f\n", chunk, end_time - start_time);


start_time = omp_get_wtime();
chunk = 27;
#pragma omp parallel for schedule(guided, chunk) num_threads(8)
for (int i = 1; i < SIZE - 1; i++) {
b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
}
end_time = omp_get_wtime();
printf("[GUIDED WITH %d CHUNK] : %2.15f\n", chunk, end_time - start_time);

start_time = omp_get_wtime();
#pragma omp parallel for schedule(auto) num_threads(8)
for (int i = 1; i < SIZE - 1; i++) {
b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
}
end_time = omp_get_wtime();
printf("[AUTO] : %2.15f\n", end_time - start_time);

start_time = omp_get_wtime();
#pragma omp parallel for schedule(runtime) num_threads(8)
for (int i = 1; i < SIZE - 1; i++) {
b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
}
end_time = omp_get_wtime();
printf("[RUNTIME] : %2.15f\n", end_time - start_time);

delete[]a, b;
}

