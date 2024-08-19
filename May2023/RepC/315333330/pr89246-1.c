#pragma omp declare simd
int foo (__int128 x)
{
return x;
}
#pragma omp declare simd
extern int bar (int x);
int
main ()
{
return foo (0) + bar (0);
}
