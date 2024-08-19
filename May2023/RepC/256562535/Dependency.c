#include <math.h>
#include <omp.h>
#include <string.h>
void a(int **a, int b) {
for (int i = 0; i < 4; ++i) {
for (int j = 1; j < 4; ++j) {
a[i + 2][j - 1] = b * a[i][j] + 4;
}
}
}
void a_sol(int **a, int b) {
int **a2;
memcpy(a2, a, sizeof(a));
#pragma omp parallel for
for (int i = 0; i < 4; ++i) {
for (int j = 1; j < 4; ++j) {
a[i + 2][j - 1] = b * a2[i][j] + 4;
}
}
}