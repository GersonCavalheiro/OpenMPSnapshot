#include <assert.h>
#pragma omp target device (cuda)
__global__ void addOne_gpu(int *a)
{
*a += 2;
}
#pragma omp target device (cuda) copy_deps
#pragma omp task inout (*a)
void addOne (int *a)
{
struct dim3 x1,x2;
x1.x = 1;
x1.y = 1;
x1.z = 1;
x2.x = 1;
x2.y = 1;
x2.z = 1;
addOne_gpu <<<x1, x2>>> (a);
}
int main (int argc, char *argv[])
{
int a = 1;
addOne(&a);
#pragma omp taskwait
assert(a == 3);
}
