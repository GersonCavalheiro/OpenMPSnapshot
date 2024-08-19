#include <math.h>
#include <omp.h>
#include <stdio.h>

const int BASE = 2;

int calc(int degree) {
return pow(BASE, degree);
}

int main() {

const int ARR_SIZE = 30;

int a[ARR_SIZE] = {1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1};

int num = 0;
#pragma omp parallel for reduction(+: num) num_threads(8)
for(int i = 0; i < ARR_SIZE; i++) {
if(a[i] != 0) {
num += calc(ARR_SIZE - i -1);
}
}

printf("Binary representation: ");
for(int n : a) {
printf("%d", n);
}
printf("\n");
printf("Decimal representation: %d\n", num);
}
