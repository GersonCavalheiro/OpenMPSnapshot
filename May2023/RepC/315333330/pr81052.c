int
foo (int x, int y)
{
int i;
#pragma omp simd
for (i = x; i < y; ++i)
return 0;			
return 1;
}
#ifdef __cplusplus
template <typename T>
T
bar (T x, T y)
{
T i;
#pragma omp simd
for (i = x; i < y; ++i)
return 0;			
return 1;
}
int x = bar (1, 7);
#endif
