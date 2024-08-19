template <typename T>
struct C
{
int foo (T n) const
{
#pragma omp parallel shared (foo)	
;
#pragma omp parallel private (foo)	
;
#pragma omp parallel firstprivate (foo)	
;
#pragma omp parallel for lastprivate (foo)	
for (T i = 0; i < n; i++)
;
#pragma omp parallel for linear (foo)	
for (T i = 0; i < n; i++)
;
#pragma omp parallel reduction (+:foo)	
;
return 0;
}
int foo (int x, int y) { return x; }
};
struct D
{
typedef int T;
int foo (T n) const
{
#pragma omp parallel shared (foo)	
;
#pragma omp parallel private (foo)	
;
#pragma omp parallel firstprivate (foo)	
;
#pragma omp parallel for lastprivate (foo)	
for (T i = 0; i < n; i++)
;
#pragma omp parallel for linear (foo)	
for (T i = 0; i < n; i++)
;
#pragma omp parallel reduction (+:foo)	
;
return 0;
}
int foo (int x, int y) { return x; }
};
int
main ()
{
C<int> ().foo (1);
D ().foo (1);
}
