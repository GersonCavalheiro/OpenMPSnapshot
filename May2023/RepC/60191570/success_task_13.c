#include<assert.h>
const int n = 10;
struct D
{
int* z;
};
int main()
{
int v[n];
for (int i = 0; i < n; ++i)
v[i] = -1;
struct D d;
d.z = v;
#pragma omp task inout(d.z[0;n]) firstprivate(n)
{
for (int i = 0; i < n; ++i)
d.z[i] = i;
}
#pragma omp taskwait
for (int i = 0; i < n; ++i)
assert(d.z[i] == i);
return 0;
}
