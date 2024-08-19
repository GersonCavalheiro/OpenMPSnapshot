int a[] = {};
void foo()
{
#pragma omp task depend(out: a[:])	
{}
}
