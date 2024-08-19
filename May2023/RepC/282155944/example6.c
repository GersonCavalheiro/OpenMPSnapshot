#include <stdio.h>
#include <omp.h>
int main(void) {
int A[4];
int val;
int sum = 0;
int i;
#pragma omp parallel num_threads(4) private(val)
{
val = omp_get_thread_num() * 10;
A[omp_get_thread_num()] = val;
}
for(i = 0; i < 4; i++){
sum += A[i];
}
printf("Sum: %d\n", sum);
return 0;
}