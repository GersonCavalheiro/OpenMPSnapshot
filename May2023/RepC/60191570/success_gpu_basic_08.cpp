#include <assert.h>
#pragma omp target device(cuda)
__global__ void addOne_gpu(int *a);
__global__ void addOne_gpu(int *a)
{
*a += 1;
}
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
}
