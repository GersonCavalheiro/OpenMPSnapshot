extern int&& ir;
#pragma omp threadprivate (ir)
int&& ir = 42;
void f()
{
ir = 24;
}
