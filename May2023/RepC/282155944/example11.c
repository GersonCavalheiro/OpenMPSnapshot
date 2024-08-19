#include <stdio.h>
#include <omp.h>
int main(void) {
int A[] = {7, 2, 3, 4};
int max = 0;
int i;
#pragma omp parallel for num_threads(2)
for(i = 0; i< 4; i++) {
#pragma omp critical
if (A[i] > max)
{
max = A[i];
}
}
printf("Max: %d\n", max);
return 0;
}