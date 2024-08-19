#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RESOLUTION_WIDTH 50
#define RESOLUTION_HEIGHT 50
#define HEAT_SOURCE_TEMP 333.00f  
#define DELTA_TIME 0.02f
#define PERROR fprintf(stderr, "%s:%d: error: %s\n", __FILE__, __LINE__, strerror(errno))
#define PERROR_GOTO(label) \
do {                   \
PERROR;            \
goto label;        \
} while (0)
#define IND(y, x) ((y) * (N) + (x))
#define SWAP(x, y)            \
do {                      \
__typeof__(x) _x = x; \
__typeof__(y) _y = y; \
x = _y;               \
y = _x;               \
} while (0)
#define FLOAT_EQUALS(x, y) fabs(x - y) < 0.01
void printTemperature(double *m, int N, int M);
int main(int argc, char **argv) {
int N = 200;
if (argc > 1) {
N = atoi(argv[1]);
}
int T = N * 10;
printf("Computing heat-distribution for room size %dX%d for T=%d timesteps\n", N, N, T);
double *A = malloc(sizeof(double) * N * N);
if (!A)
PERROR_GOTO(error_a);
double *B = malloc(sizeof(double) * N * N);
if (!B)
PERROR_GOTO(error_b);
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
A[IND(i, j)] = 273;  
B[IND(i, j)] = 273;
}
}
int source_x = N / 4;
int source_y = N / 4;
A[IND(source_x, source_y)] = HEAT_SOURCE_TEMP;
B[IND(source_x, source_y)] = HEAT_SOURCE_TEMP;
printf("Initial:");
printTemperature(A, N, N);
printf("\n");
for (int t = 0; t < T; t++) {
#pragma omp parallel for collapse(2)
for (int i = 1; i < N - 1; i++) {
for (int j = 1; j < N - 1; j++) {
double l, r, u, d;
if (IND(source_x, source_y) != IND(i, j)) {
l = A[IND(i - 1, j)];
r = A[IND(i + 1, j)];
u = A[IND(i, j - 1)];
d = A[IND(i, j + 1)];
B[IND(i, j)] = ((l + r + u + d) / 4 - A[IND(i, j)]) * DELTA_TIME + A[IND(i, j)];
}
}
}
if (!FLOAT_EQUALS(B[IND(source_x, source_y)], HEAT_SOURCE_TEMP)) {
fprintf(stderr, "Error: Heat source changed! Source heat = %.2fF\n", B[IND(source_x, source_y)]);
errno = ECANCELED;
PERROR_GOTO(error_b);
}
SWAP(A, B);
if (!(t % 1000)) {
printf("Step t=%d\n", t);
printTemperature(A, N, N);
printf("\n");
}
}
printf("Final:");
printTemperature(A, N, N);
printf("\n");
int success = 1;
for (long long i = 0; i < N; i++) {
for (long long j = 0; j < N; j++) {
double temp = A[IND(i, j)];
if (273 <= temp && temp <= 273 + 60)
continue;
success = 0;
break;
}
}
printf("Verification: %s\n", (success) ? "OK" : "FAILED");
error_b:
free(B);
error_a:
free(A);
return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
}
void printTemperature(double *m, int N, int M) {
const char *colors = " .-:=+*^X#%@";
const int numColors = 12;
const double max = 273 + 30;
const double min = 273 + 0;
int W = RESOLUTION_WIDTH;
int H = RESOLUTION_HEIGHT;
int sW = N / W;
int sH = M / H;
printf("\t");
for (int u = 0; u < W + 2; u++) {
printf("X");
}
printf("\n");
for (int i = 0; i < H; i++) {
printf("\tX");
for (int j = 0; j < W; j++) {
double max_t = 0;
for (int x = sH * i; x < sH * i + sH; x++) {
for (int y = sW * j; y < sW * j + sW; y++) {
max_t = (max_t < m[IND(x, y)]) ? m[IND(x, y)] : max_t;
}
}
double temp = max_t;
int c = ((temp - min) / (max - min)) * numColors;
c = (c >= numColors) ? numColors - 1 : ((c < 0) ? 0 : c);
printf("%c", colors[c]);
}
printf("X\n");
}
printf("\t");
for (int l = 0; l < W + 2; l++) {
printf("X");
}
printf("\n");
}
