#include <stdio.h>
#include <omp.h>
int main(int argc, char* argv[]) {
int i, n = 10, sum = 0, last_sum = 0;
#pragma omp parallel for lastprivate(last_sum)
for (i = 0; i < n; i++) {
last_sum = sum;
sum += i;
printf("Thread %d: last_sum = %d, sum = %d\n", omp_get_thread_num(), last_sum, sum);
}
printf("Final sum: %d\n", sum);
return 0;
}
