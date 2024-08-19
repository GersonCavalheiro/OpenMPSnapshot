#include "iostream"
#include "omp.h"

int main() {
int a = 0, b = 0;

printf("[Before entering into 1 area] : a = %d, b = %d\n", a, b);
#pragma omp parallel private(a) firstprivate(b) num_threads(2)
{
int thread_id = omp_get_thread_num();
a += thread_id;
b += thread_id;
printf("[In 1 area] : a = %d, b = %d\n", a, b);
}
printf("[After 1 area] : a = %d, b = %d\n\n", a, b);

printf("[Before entering into 2 area] : a = %d, b =%d\n", a, b);
#pragma omp parallel shared(a) private(b) num_threads(4)
{
int thread_id = omp_get_thread_num();
a -= thread_id;
b -= thread_id;
printf("[In 2 area] : a = %d, b = %d\n", a, b);
}
printf("[After 2 area] : a = %d, b = %d\n", a, b);

}
