void bar (void);
int foo (int, int) __attribute__((used));
#pragma omp declare simd uniform (b) linear(a:b)
int
foo (int a, int b)
{
a = a + 1;
#pragma omp parallel
bar ();  
return a;
}
int foo (int, int)  __attribute__((unused));
