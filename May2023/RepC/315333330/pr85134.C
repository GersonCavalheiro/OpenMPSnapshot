void
foo (int i)
{
constexpr int x[i] = {};	
#pragma omp parallel shared(x)
;
}
