#include <zconf.h>
#include "iostream"
#include "omp.h"
#include "../../array_utils.h"

const int ARRAY_SIZE = 10;

int main() {
int *a = new int[ARRAY_SIZE];
int *b = new int[ARRAY_SIZE];

srand(time(NULL));

for (int i = 0; i < ARRAY_SIZE; i++) {
a[i] = rand() % 10 - 5;
b[i] = rand() % 10 - 5;
}

print_array(a, ARRAY_SIZE);
printf("\n");
print_array(b, ARRAY_SIZE);
printf("\n");

int max = -5, min = 5;

#pragma omp parallel sections num_threads(2)
{
#pragma omp section
{
printf("search MIN %d\n", omp_get_thread_num());
for (int i = 0; i < ARRAY_SIZE; i++) {
if (a[i] < min) {
min = a[i];
}
}
sleep(1);
}
#pragma omp section
{
printf("search MAX %d\n", omp_get_thread_num());
for (int i = 0; i < ARRAY_SIZE; i++) {
if (b[i] > max) {
max = b[i];
}
}
}
}

printf("min : %d, max : %d", min, max);

delete[] a, b;
}

