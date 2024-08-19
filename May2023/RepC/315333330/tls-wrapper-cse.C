int f() { return 42; }
int ir = f();
#pragma omp threadprivate (ir)
int main()
{
return ir + ir - 84;
}
