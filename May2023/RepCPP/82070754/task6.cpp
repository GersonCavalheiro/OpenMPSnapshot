#include "iostream"
#include "omp.h"

int ARRAY_SIZE = 10;

int main() {

int *a = new int[ARRAY_SIZE];
int *b = new int[ARRAY_SIZE];

srand(time(NULL));

for (int i = 0; i < ARRAY_SIZE; i++) {
a[i] = rand() % 100 - 50;
b[i] = rand() % 100 - 50;
}

int sum_a = 0, sum_b = 0;
double avg_a, avg_b;

#pragma omp parallel for reduction(+:sum_a, sum_b)
for (int i = 0; i < ARRAY_SIZE; i++) {
sum_a += a[i];
sum_b += b[i];
}

avg_a = (double) sum_a / ARRAY_SIZE;
avg_b = (double) sum_b / ARRAY_SIZE;
printf("%d, %d\n", sum_a, sum_b);
printf("%4.2f , %4.2f", avg_a, avg_b);
delete[] a, b;
}
