#include "iostream"
#include "omp.h"
#include "../../array_utils.h"

const int SIZE = 30;

int main() {

srand(time(NULL));

int *arr = new int[SIZE];
for (int i = 0; i < SIZE; i++) {
arr[i] = rand() % 10;
}

print_array(arr, SIZE);
printf("\n");

int n = 0;


#pragma omp parallel for num_threads(10)
for (int i = 0; i < SIZE; i++) {
if (arr[i] % 3 == 0) {
#pragma omp atomic
n++;
}
}

cout << n << endl;
}
