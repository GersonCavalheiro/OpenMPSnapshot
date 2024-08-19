void foo(int* x, int global, int local);
int main()
{
int x[2];
#pragma omp target device(opencl) copy_deps ndrange(1, global, local) file(dummy.cl)
#pragma omp task in([global]x)
void foo(int* x, int global, int local);
foo(x, 2, 1);
#pragma omp taskwait
}
