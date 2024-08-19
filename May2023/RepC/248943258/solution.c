#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
double Func(double x) {
if (x > 2) {
return 0;
}
return sqrt(4 - x*x);
}
double Integral(size_t left_index, size_t right_index, double h) {
double I = (Func(right_index * h) + Func(left_index * h)) / 2;
for (size_t i = left_index + 1; i < right_index; i++) {
I += Func(i * h);
}
return I * h;
}
int main(int argc, char **argv) {
size_t N = 1000000;
int size = 1;
size_t numexp = 1;
if (argc > 1) {
N = atoll(argv[1]);
if (argc > 2) {
size = atoi(argv[2]);
if (argc > 3) {
numexp = atoll(argv[3]);
}
}
}
double a = 0, b = 2;
double h = (b - a) / N;
double result = 0.0;
for (size_t i = 0; i < numexp; i++) {
omp_set_num_threads(size);
#pragma omp parallel
{
int rank = omp_get_thread_num();
size_t left_index = rank * (N / size);
size_t right_index =
(rank != size - 1) ? (rank + 1) * (N / size) : N;
double integral = Integral(left_index, right_index, h);
#pragma omp critical
result += integral;
}
}
printf("%d %lf\n", size, result / numexp);
return EXIT_SUCCESS;
}
