#include <assert.h>
#pragma omp target device(cuda)
__global__ void addOne_gpu(int *a)
{
*a += 1;
}
struct MyType
{
int x;
};
#pragma omp target device (cuda) copy_deps
#pragma omp task inout (*a)
void addOne (struct MyType *a)
{
addOne_gpu <<<1, 1>>> (&(a->x));
}
int main (int argc, char *argv[])
{
struct MyType var;
var.x = 1;
addOne(&var);
#pragma omp taskwait
assert(var.x == 2);
}
