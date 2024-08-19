#pragma omp declare target
void f1 (int);
void f1 (double);
template <typename T>
void f2 (T);
template<> void f2<int> (int);
#pragma omp end declare target
void f3 (int);
void f4 (int);
void f4 (short);
template <typename T>
void f5 (T);
#pragma omp declare target (f3)
#pragma omp declare target to (f4)	
#pragma omp declare target to (f5<int>)	
template <int N>
void f6 (int)
{
static int s;
#pragma omp declare target (s)
}
namespace N
{
namespace M
{
void f7 (int);
}
void f8 (long);
}
void f9 (short);
int v;
#pragma omp declare target (N::M::f7)
#pragma omp declare target to (::N::f8)
#pragma omp declare target to (::f9) to (::v)
