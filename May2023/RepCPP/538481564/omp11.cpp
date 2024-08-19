#include <limits>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

int main() {

const int ARR_SIZE = 30;
const int DIVIDER = 7;
const int LIMIT = 100;

int max = numeric_limits<int>::min();

srand(time(NULL));

int a[ARR_SIZE];

for (int i = 0; i < ARR_SIZE; i++){
a[i] = rand() % LIMIT;
printf("%d ", a[i]);
}
printf("\n");

#pragma omp parallel for num_threads(8)
for (int i = 0; i < ARR_SIZE; i++) {

if (a[i] % DIVIDER == 0 && a[i] > max) {

#pragma omp critical
{
if (a[i] % DIVIDER == 0 && a[i] > max) {
max = a[i];
}
}
}
}

if (max != numeric_limits<int>::min()) {
printf("Max multiple of 7 value in the array: %d\n", max);
}
else {
printf("There is no multiples of 7 values in the array\n");
}
}
