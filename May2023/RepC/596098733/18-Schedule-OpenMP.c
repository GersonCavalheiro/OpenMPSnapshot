#include <stdio.h>
#include <omp.h>
#define N 10
int main() {
int i;
#pragma omp parallel for schedule(static, 2)
for (i = 0; i < N; i++) {
printf("Thread %d processing element %d with static scheduling\n", omp_get_thread_num(), i);
}
#pragma omp parallel for schedule(dynamic, 2)
for (i = 0; i < N; i++) {
printf("Thread %d processing element %d with dynamic scheduling\n", omp_get_thread_num(), i);
}
#pragma omp parallel for schedule(guided, 2)
for (i = 0; i < N; i++) {
printf("Thread %d processing element %d with guided scheduling\n", omp_get_thread_num(), i);
}
#pragma omp parallel for schedule(runtime)
for (i = 0; i < N; i++) {
printf("Thread %d processing element %d with runtime scheduling\n", omp_get_thread_num(), i);
}
return 0;
}
