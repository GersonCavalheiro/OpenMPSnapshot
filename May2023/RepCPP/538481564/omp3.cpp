#include <omp.h>
#include <stdio.h>

int main() {

int a = 0, b = 0;

printf("Before increasing: a = %d, b = %d\n", a, b);

#pragma omp parallel num_threads(2) private(a) firstprivate(b)
{
int nthread = omp_get_thread_num();

a += nthread;
b += nthread;
printf("During increasing: a = %d, b = %d (on thread num %d)\n", a, b, nthread);
}
printf("After increasing: a = %d, b = %d\n", a, b);

printf("Before decreasing: a = %d, b = %d\n", a, b);

#pragma omp parallel num_threads(4) shared(a) private(b)
{
int nthread = omp_get_thread_num();

a -= nthread;
b -= nthread;
printf("During decreasing: a = %d, b = %d (on thread num %d)\n", a, b, nthread);
}
printf("After decreasing: a = %d, b = %d\n", a, b);
}
