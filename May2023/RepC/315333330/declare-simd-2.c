#pragma omp declare simd
int a;	
#pragma omp declare simd
int fn1 (int a), fn2 (int a);	
#pragma omp declare simd
int b, fn3 (int a);	
#pragma omp declare simd linear (a)
int fn4 (int a), c;	
#pragma omp declare simd
extern "C"		
{
int fn5 (int a);
}
#pragma omp declare simd 
namespace N1
{
int fn6 (int a);
}
#pragma omp declare simd simdlen (4)
struct A
{			
int fn7 (int a);
};
#pragma omp declare simd
template <typename T>
struct B
{			
int fn8 (int a);
};
struct C
{
#pragma omp declare simd 
public:		 
int fn9 (int a);
};
int t;
#pragma omp declare simd
#pragma omp declare simd
#pragma omp threadprivate(t)	
int fn10 (int a);
#pragma omp declare simd inbranch notinbranch 
int fn11 (int);
struct D
{
int d;
#pragma omp declare simd simdlen (N) linear (a : sizeof (e) + sizeof (this->e)) 
template <int N>
int fn12 (int a);
int e;
};
#pragma omp declare simd aligned (a, b, c, d)
int fn13 (int *a, int b[64], int *&c, int (&d)[64]);
#pragma omp declare simd aligned (a)	
int fn14 (int a);
#pragma omp declare simd aligned (b)	
int fn14 (int &b);
#pragma omp declare simd aligned (c)	
int fn14 (float c);
#pragma omp declare simd aligned (d)	
int fn14 (double &d);
#pragma omp declare simd aligned (e)	
int fn14 (D e);
#pragma omp declare simd linear(a:7) uniform(a)	
int f15 (int a);
#pragma omp declare simd linear(a) linear(a)	
int f16 (int a);
#pragma omp declare simd linear(a) linear(a:7)	
int f17 (int a);
#pragma omp declare simd linear(a:6) linear(a:6)
int f18 (int a);
#pragma omp declare simd uniform(a) uniform(a)	
int f19 (int a);
#pragma omp declare simd uniform(a) aligned (a: 32)
int f20 (int *a);
