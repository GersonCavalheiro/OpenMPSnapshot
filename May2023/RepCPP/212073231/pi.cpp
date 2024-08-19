#include <iostream>
#include <cmath>

static long num_steps = 50000;
double step;

int pi() {
int i;
double x, pi, sum = 0.0, d;
step = 1.0 / (double) num_steps;

#pragma omp parallel for reduction(+ : sum) private(i, x, d) shared(step), default(none)
for (i = 1; i <= (int) num_steps; i++) {
x = (i - 0.5) * step;
d = step * sqrt(1.0 - x * x);
sum = sum + d;
}

pi = 4 * sum;

printf("%f", pi);
}