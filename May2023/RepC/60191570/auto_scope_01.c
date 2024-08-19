int fibonacci(int n)
{
long long x, y;
if (n < 2) return n;
#pragma analysis_check assert auto_sc_firstprivate(n) auto_sc_shared(x)
#pragma omp task default(AUTO)
x = fibonacci(n - 1);
#pragma analysis_check assert auto_sc_firstprivate(n) auto_sc_shared(y)
#pragma omp task default(AUTO) 
y = fibonacci(n - 2);
#pragma omp taskwait
return x + y;
}