#include <assert.h>
#include "success_gpu_basic_02.cu"
#pragma omp target device (cuda) copy_deps
#pragma omp task inout (*a)
void addOne (int *a)
{
addOne_gpu <<<1, 1>>> (a);
}
int main (int argc, char *argv[])
{
int a = 1;
addOne(&a);
#pragma omp taskwait
assert(a == 2);
return 0;
}
