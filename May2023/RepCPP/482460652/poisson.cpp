#include <omp.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef std::vector<double> Vector;

double func(double x, double y) {
return sin(x) * cos(y);
}

double right(double x, double y) {
return -2 * sin(x) * cos(y);
}

int main(int argc, char **argv) {
const int N = (argc > 1 ? atoi(argv[1]) : 1000);
const double eps = (argc > 2 ? atof(argv[2]) : 1e-8);
const int max_iter = (argc > 3 ? atoi(argv[3]) : 1000);
const double h = 1.0 / N;
Vector A(N * N), A_next(N * N), f(N * N);

#pragma omp parallel for
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
if (i == 0 || j == 0 || i == N - 1 || j == N - 1) {
A[i * N + j] = func(i * h, j * h);
} else {
A[i * N + j] = 0.0;
}
f[i * N + j] = right(i * h, j * h);
}
}

double norm = 0.0;
int iter = 0;
do {
norm = 0.0;
#pragma omp parallel for reduction(max:norm) schedule(dynamic, 100)
for (int i = 1; i < N - 1; i++) {
for (int j = 1; j < N - 1; j++) {
A_next[i * N + j] = (A[(i - 1) * N + j] + A[(i + 1) * N + j]
+ A[i * N + j - 1] + A[i * N + j + 1]
- h * h * f[i * N + j]) / 4.0;
norm = fmax(norm, fabs(A_next[i * N + j] - A[i * N + j]));
}
}
#pragma omp parallel for schedule(dynamic, 100)
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
A[i * N + j] = A_next[i * N + j];
}
}
iter++;
} while ((norm > eps) && (iter < max_iter));

std::cout << "Norm = " << norm << ", iter = " << iter << std::endl;
return 0;
}
