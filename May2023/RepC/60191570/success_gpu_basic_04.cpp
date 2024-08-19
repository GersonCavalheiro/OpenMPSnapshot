#include <assert.h>
#include "success_gpu_basic_04.cu"
#pragma omp target device (cuda) copy_deps
#pragma omp task inout (*a)
void addOne (struct MyType *a)
{
addOne_gpu <<<1, 1>>> (&(a->x));
}
int main (int argc, char *argv[])
{
MyType var;
var.x = 1;
addOne(&var);
#pragma omp taskwait
assert(var.x == 2);
}
