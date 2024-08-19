#include <stdio.h>
#include <omp.h>
int main() {
#pragma omp parallel num_threads(8)
{
printf("hello i'm %d\n", omp_get_thread_num());
}
return 0;
}
