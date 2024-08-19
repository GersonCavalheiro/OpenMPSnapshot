struct S { int n; };
#pragma omp declare simd
struct S
foo (int x)
{
return (struct S) { x };
}
#pragma omp declare simd
int
bar (struct S x)
{
return x.n;
}
#pragma omp declare simd uniform (x)
int
baz (int w, struct S x, int y)
{
return w + x.n + y;
}
