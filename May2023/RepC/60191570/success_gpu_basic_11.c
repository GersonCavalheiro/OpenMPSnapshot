#pragma omp task
void foo_smp()
{
}
#pragma omp target device(cuda)
__global__ void foo_gpu()
{
}
#pragma omp target device(cuda) implements(foo_smp)
#pragma omp task
void foo_gpu_wrapper()
{
foo_gpu<<<1,1>>>();
}
inline void f()
{
foo_smp();
}
int main()
{
f();
#pragma omp taskwait
}
