#include "iostream"
#include "omp.h"
#include "../../array_utils.h"

const int SIZE = 20;

int main() {

srand(time(NULL));

int *arr = new int[SIZE];
for (int i = 0; i < SIZE; i++) {
arr[i] = rand() % 10;
}

print_array(arr, SIZE);
printf("\n");

int n = 0;

#pragma omp parallel for num_threads(20)
for (int j = 0; j < SIZE; ++j) {
if (arr[j] % 7 == 0) {
#pragma omp critical
n++;
}
}

printf("%d", n);

}

