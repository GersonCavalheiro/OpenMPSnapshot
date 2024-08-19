#include <omp.h>
#include <stdio.h>

const int ARR_SIZE = 12;

void print_array(int arr[ARR_SIZE]){

for (int i = 0; i < ARR_SIZE; i++) {
printf("%d ", arr[i]);
} 
printf("\n");
}

int main() {

int a[ARR_SIZE], b[ARR_SIZE], c[ARR_SIZE];

omp_set_num_threads(3);
printf("Number of threads in first cycle: %d\n", omp_get_max_threads());
#pragma omp parallel for schedule(static, 3)
for (int i = 0; i < ARR_SIZE; i++) {
a[i] = i;
b[i] = i * i;
printf("Thread %d out of %d (on index %d)\n", omp_get_thread_num(), omp_get_num_threads(), i);
}
print_array(a);
print_array(b);

omp_set_num_threads(4);
printf("Number of threads in second cycle: %d\n", omp_get_max_threads());
#pragma omp parallel for schedule(dynamic, 3)
for (int i = 0; i < ARR_SIZE; i++) {
c[i] = a[i] + b[i];
printf("Thread %d out of %d (on index %d)\n", omp_get_thread_num(), omp_get_num_threads(), i);
}
print_array(c);
}
