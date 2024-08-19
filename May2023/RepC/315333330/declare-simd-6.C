struct S { int a; };
#pragma omp declare simd linear(val(a):2)
int f1 (int &a);
#pragma omp declare simd linear(uval(a):2)
unsigned short f2 (unsigned short &a);
#pragma omp declare simd linear(ref(a):1)
int f3 (long long int &a);
#pragma omp declare simd linear(a:1)
int f4 (int &a);
#pragma omp declare simd linear(val(a))
int f5 (int a);
#pragma omp declare simd linear(uval(a):2)		
int f6 (unsigned short a);
#pragma omp declare simd linear(ref(a):1)		
int f7 (unsigned long int a);
#pragma omp declare simd linear(a:1)
int f8 (int a);
#pragma omp declare simd linear(val(a):2)		
int f9 (S &a);
#pragma omp declare simd linear(uval(a):2)		
int f10 (S &a);
#pragma omp declare simd linear(ref(a):1)		
int f11 (S &a);
#pragma omp declare simd linear(a:1)			
int f12 (S &a);
#pragma omp declare simd linear(val(a))			
int f13 (S a);
#pragma omp declare simd linear(uval(a):2)		
int f14 (S a);
#pragma omp declare simd linear(ref(a):1)		
int f15 (S a);
#pragma omp declare simd linear(a:1)			
int f16 (S a);
