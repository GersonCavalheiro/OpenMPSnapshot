





#pragma omp target device(smp) copy_deps
#pragma omp task out(*a)
void foo(float* a)
{
}

int main ()
{
float *a;
foo(a);
foo(a);
#pragma omp taskwait
return 0;
}
