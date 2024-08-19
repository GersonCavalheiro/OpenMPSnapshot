
#include <cstdio>
#include <cmath>
#include "omp.h"


double integralFunction(double x);


double calculatePI(double down, double up, long numSteps, int numberOfThreads = 1);


double parallelBlock(long newNumSteps, double threadDown, double threadUp);

int main() {

printf("Running on a system with %d threads\n\n", omp_get_max_threads());

int i = 1;
while (true) {
double pi = calculatePI(0.0, 1.0, 1'000'000'000L, i++);

printf("pi = %.20f\n", pi);

if (i == omp_get_max_threads() + 1)
break;
}
}

double integralFunction(double x) {
return 4.0 / (1.0 + x * x);
}

double calculatePI(double down, double up, long numSteps, int numberOfThreads) {

double sum = 0.0;
double start = omp_get_wtime();
#pragma omp parallel num_threads(numberOfThreads)
{
double threadDown = (double) omp_get_thread_num() / (double) numberOfThreads;
double threadUp = (double) (omp_get_thread_num() + 1) / (double) numberOfThreads;
double temp = parallelBlock(numSteps / numberOfThreads, threadDown, threadUp);
#pragma omp critical
sum += temp;
}
double end = omp_get_wtime();
printf("%2d threads in %.8f seconds\t", numberOfThreads, end - start);

return sum;
}

double parallelBlock(long newNumSteps, double threadDown, double threadUp) {
double sum = 0.0;

double dx = (threadUp - threadDown) / (double) newNumSteps;

for (long i = 0; i < newNumSteps; i++) {
double x = (i + 0.5) * dx;
x += threadDown;
double funRes = integralFunction(x);
sum += funRes * dx;
}
return sum;
}
