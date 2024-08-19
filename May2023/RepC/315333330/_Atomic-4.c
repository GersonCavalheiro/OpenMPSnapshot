#pragma omp declare simd
int
f1 (_Atomic int x, int y)	
{
return x + y;
}
#pragma omp declare simd uniform(x)
int
f2 (_Atomic int x, int y)
{
return x + y;
}
