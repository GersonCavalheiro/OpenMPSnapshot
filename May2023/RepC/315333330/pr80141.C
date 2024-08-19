#pragma omp declare simd aligned (p : 2 && 2)
template<int> void foo (int *p);
#pragma omp declare simd simdlen (2 && 2)
template<int> void bar (int *p);
