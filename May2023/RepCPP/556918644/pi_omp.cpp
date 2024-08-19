
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <math.h>
#include <omp.h>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define THREADS 12

int main(int argc, char **argv)
{
srand(time(NULL));

int sum = 0;

int N = 256 * 256 * atoi(argv[1]);

double x, y;

auto start = high_resolution_clock::now();

#pragma omp parallel for private(x, y) reduction(+ : sum) num_threads(THREADS)
for (int i = 0; i < N; i++)
{
x = (double)rand() / RAND_MAX;
y = (double)rand() / RAND_MAX;

if (x * x + y * y <= 1.0)
sum++;
}

auto stop = high_resolution_clock::now();

double pi = sum * 4.0 / N;

double err = abs(pi - acos(-1)) / pi * 100;


duration<double, std::milli> ms_double = stop - start;

printf("\nCompleted successfully!\n");
printf("CPU PI = %f\n", pi);
printf("CPU relative error = %f pct\n", err);
printf("calculatePi() execution time on the CPU: %f ms\n", ms_double.count());

return 0;
}