template <int N>
int
foo (int *v, int A, int B)	
{
int r = 0;
int a = 2;			
int b = 4;			
#pragma omp target map(to: v[a:b])
r |= v[3];
#pragma omp target map(to: v[A:B])
r |= v[3];
return r;
}
template <typename T>
int
bar (T *v, T A, T B)		
{
T r = 0, a = 2, b = 4;	
#pragma omp target map(to: v[a:b])
r |= v[3];
#pragma omp target map(to: v[A:B])
r |= v[3];
return r;
}
int
baz (int *v, int A, int B)
{
return foo<0> (v, A, B) + bar (v, A, B);
}
