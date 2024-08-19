#include <stdio.h>
#include <omp.h>
int main() {
int x = 0;
#pragma omp parallel firstprivate(x)
{
int id = omp_get_thread_num();
x += id;
printf("Thread %d: x = %d\n", id, x);
}
printf("After parallel region: x = %d\n", x);
return 0;
}
