#include<assert.h>
#include<unistd.h>
#pragma omp task inout(v[0:n-1])
template < typename T>
void foo_array_section_range(T *v, int n)
{
sleep(1);
for (int i = 0; i < n; ++i)
{
v[i] += i;
}
}
#pragma omp task inout(v[0;n])
template < typename T>
void foo_array_section_size(T *v, int n)
{
sleep(1);
for (int i = 0; i < n; ++i)
{
v[i] += i;
}
}
int main()
{
const int n = 10;
int v[n];
#pragma omp task out(v)
{
for (int i = 0; i < n; ++i)
v[i] = 0;
}
foo_array_section_range(v, n);
#pragma omp task in(v)
{
for (int i = 0; i < n; ++i)
assert(v[i] == i);
}
foo_array_section_size(v, n);
#pragma omp task in(v)
{
for (int i = 0; i < n; ++i)
assert(v[i] == 2*i);
}
#pragma omp taskwait
}
