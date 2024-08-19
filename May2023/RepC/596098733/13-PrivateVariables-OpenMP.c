#include <stdio.h>
#include <omp.h>
#define N 10
int main(int argc, char *argv[]) {
int i;
int sum = 0;
#pragma omp parallel private(i) reduction(+:sum)
{
#pragma omp for
for (i = 0; i < N; i++) {
int tid = omp_get_thread_num();
int local_sum = i * tid;
sum += local_sum;
}
}
printf("The sum is %d\n", sum);
return 0;
}
