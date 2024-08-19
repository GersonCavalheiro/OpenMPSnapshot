void bar (void);
#pragma omp declare simd uniform (b) linear(a:b)
int
foo (int a, int b)
{
a = a + 1;
#pragma omp parallel
bar ();  
return a;
}
