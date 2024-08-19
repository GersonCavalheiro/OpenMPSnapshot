#include<assert.h>
const int  n = 10;
struct D
{
int z;
};
int main()
{
int v[n];
struct D d[n];
for (int i = 0; i < n; ++i)
d[i].z = -1;
#pragma omp task inout(d[0;n]) firstprivate(n)
{
for (int i = 0; i < n; ++i)
d[i].z = i;
}
#pragma omp taskwait
for (int i = 0; i < n; ++i)
assert(d[i].z == i);
return 0;
}
