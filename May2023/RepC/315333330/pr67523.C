struct S { int s; };
template <typename T>
void foo (T &x, T &y)
{
#pragma omp for simd
for (T i = x; i < y; i++)	
;
#pragma omp parallel for simd
for (T i = x; i < y; i++)	
;
#pragma omp target teams distribute parallel for simd
for (T i = x; i < y; i++)	
;
#pragma omp target teams distribute simd
for (T i = x; i < y; i++)	
;
}
void
bar ()
{
S x, y;
foo <S> (x, y);
}
