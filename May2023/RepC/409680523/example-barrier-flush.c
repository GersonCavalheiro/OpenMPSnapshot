#include <stdio.h>
#include <omp.h>
int main(void)
{
int x;
x=2; 
#pragma omp parallel num_threads(5)
{
if (omp_get_thread_num() == 0) {
x=5; 
}
else {
printf("Thread %d x=%d before barrier\n", omp_get_thread_num(), x);
}
#pragma omp barrier
if (omp_get_thread_num() == 0) {
printf("Thread %d x=%d after barrier\n", omp_get_thread_num(), x);
}
else {
printf("Thread %d x=%d after barrier\n", omp_get_thread_num(), x);
}
}
return 0;
}
