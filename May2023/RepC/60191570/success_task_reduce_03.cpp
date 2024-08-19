#include <omp.h>
#include <assert.h>
int main(int argc, char *argv[]) {
int a[10] = {0};
#pragma omp parallel
#pragma omp single
{
#pragma omp taskgroup task_reduction(+:a)
{
for (int i = 0; i < 10; ++i)
#pragma omp task in_reduction(+:a) firstprivate(i)
{
a[i] = a[i] + i;
}
}
for (int i = 0; i < 10; ++i) {
assert(a[i] == i);
}
}
}
