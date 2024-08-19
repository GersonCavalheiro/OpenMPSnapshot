#pragma omp declare simd
extern int foo (__int128 x);
#pragma omp declare simd
int
bar (int x)
{
return x + foo (0);
}
