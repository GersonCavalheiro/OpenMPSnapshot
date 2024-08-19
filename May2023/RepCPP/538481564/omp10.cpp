#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

const int ARR_SIZE = 30;
const int DIVIDER = 9;
const int LIMIT = 100;

srand(time(NULL));

int a[ARR_SIZE];

for (int i = 0; i < ARR_SIZE; i++){

a[i] = rand() % LIMIT;
printf("%d ", a[i]);
}
printf("\n");

int count = 0;
#pragma omp parallel for num_threads(8)
for (int i = 0; i < ARR_SIZE; i++) {

if (a[i] % DIVIDER == 0) {

#pragma omp atomic
count++;
}
}
printf("Amount of multiples of 9 numbers: %d\n", count);
}
