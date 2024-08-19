#include <math.h>
#include <omp.h>
#include <stdio.h>

int main() {

const int NUMBER = 210;

int result = 0;
#pragma omp parallel for reduction(+: result) num_threads(8)
for(int i = 1; i <= NUMBER*2; i += 2) {
result += i;
}
printf("%d^%d = %d\n", NUMBER, 2, result);
}