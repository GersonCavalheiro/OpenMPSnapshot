template < typename T>
int foo(T* x, T* y,  int n)
{
int result = 0;
#pragma omp parallel for reduction(+:result)
for (int i = 0; i < n; ++i)
{
result += x[i] + y[i];
}
return result;
}
#define N 5
int main()
{
int v1[N] = {1, 2, 3, 4, 5};
int v2[N] = {5, 4, 3, 2, 1};
int res = foo(v1, v2, N);
}
