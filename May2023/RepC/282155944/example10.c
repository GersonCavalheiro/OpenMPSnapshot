#include <stdio.h>
#include <omp.h>
int main(void) {
int A[] = {1, 2, 3, 4};
int sum = 0;
int i;
#pragma omp parallel for num_threads(2) reduction(+: sum)
for(i = 0; i< 4; i++) {
sum += A[i];
}
printf("Sum: %d\n", sum);
return 0;
}