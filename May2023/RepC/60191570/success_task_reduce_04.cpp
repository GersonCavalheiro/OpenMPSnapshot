#include <omp.h>
#include <assert.h>
struct A {
int x;
};
#pragma omp declare reduction(my_add: A : omp_out.x = omp_in.x + omp_out.x) initializer(omp_priv = {0})
int main(int argc, char *argv[]) {
A a[10] = {0};
#pragma omp parallel
#pragma omp single
{
#pragma omp taskgroup task_reduction(my_add:a)
{
for (int i = 0; i < 10; ++i)
#pragma omp task in_reduction(my_add:a) firstprivate(i)
{
a[i].x = a[i].x + i;
}
}
for (int i = 0; i < 10; ++i)
{
assert(a[i].x == i);
}
}
}
