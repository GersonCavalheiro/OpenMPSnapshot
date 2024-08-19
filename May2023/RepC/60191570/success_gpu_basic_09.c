#pragma omp target device(smp)
#pragma omp task
void foo_smp()
{
}
#pragma omp target device(smp) implements(foo_smp)
#pragma omp task
void foo_smp_v2()
{
}
#pragma omp target device(cuda) implements(foo_smp)
#pragma omp task
void foo_cuda()
{
}
int main()
{
foo_smp();
#pragma omp taskwait
}
