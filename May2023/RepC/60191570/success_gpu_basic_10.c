#include <assert.h>
#pragma omp target device(cuda)
__global__ void addOne_gpu(int *a)
{
*a += 1;
}
int main (int argc, char *argv[])
{
int a = 1;
#pragma omp target device (cuda) copy_deps
#pragma omp task inout (a)
{
dim3 var;
var.x = 1;
var.y = 1;
var.z = 1;
addOne_gpu <<<1, 1>>> (&a);
}
#pragma omp taskwait
assert(a == 2);
return 0;
}
