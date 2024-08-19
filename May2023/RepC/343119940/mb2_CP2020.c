#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void mxv(int m, int n, double * a, double * b, double * c);
void main() {
int i, j, k, n, m;
float tiempo_antes, tiempo_despues;
double * c, * b, * a;
int hilos[4] = {
1,
2,
4,
8
}; 
long int Niteraciones = 9e8;
int N = 1;
for (k = 0; k < 4; k++) {
N = 10 * N;
Niteraciones = Niteraciones / 100;
n = N;
m = N;
if ((b = (double * ) malloc(m * n * sizeof(double))) == NULL) {
perror("Error al reservar memoria para b");
exit(EXIT_FAILURE);
}
if ((c = (double * ) malloc(n * sizeof(double))) == NULL) {
perror("Error al reservar memoria para c");
exit(EXIT_FAILURE);
}
for (j = 0; j < n; j++) c[j] = 2.0;
for (i = 0; i < m; i++)
for (j = 0; j < n; j++) b[i * n + j] = i;
for (j = 0; j < 4; j++) {
omp_set_num_threads(hilos[j]);
if ((a = (double * ) malloc(m * sizeof(double))) == NULL) {
perror("Error al reservar memoria para a");
exit(EXIT_FAILURE);
}
tiempo_antes = omp_get_wtime();
#pragma omp parallel for private(i)
for (i = 0; i < Niteraciones; i++) {
mxv(m, n, a, b, c);
}
tiempo_despues = omp_get_wtime();
free(a);
printf("Numero de hilos: %i", hilos[j]);
printf("\tVolumen Datos = %8.4f MB, N = %6i ", (float)(N * N + 2 * N) / (1024.0 * 1024.0), N);
printf("\tTiempo = %5.4f seg ", tiempo_despues - tiempo_antes);
float Nd = (float) N;
float Nditeraciones = (float) Niteraciones;
float tiempo = tiempo_despues - tiempo_antes;
printf("\tMFLOPs = %10.4f  \n", tiempo > 0 ? (float)(2 * Nd * Nd * Nditeraciones) / (1e6 * tiempo) : 0);
} 
free(b), free(c);
} 
}
void mxv(int m, int n, double * a, double * b, double * c) {
int i, j;
#pragma omp parallel shared (m, n, a, b, c ) private (i, j)
#pragma omp for
for (i = 0; i < m; i++) {
a[i] = 0.0;
for (j = 0; j < n; j++) a[i] += b[i * n + j] * c[j];
}
}