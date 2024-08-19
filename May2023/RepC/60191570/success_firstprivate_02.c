#include<assert.h>
int v(int n)
{
int v[n];
for (int i = 0; i < n; ++i)
v[i] = 0;
#pragma omp task firstprivate(v)
{
for (int i = 0; i < n; ++i)
v[i]++;
}
#pragma omp taskwait
for (int i = 0; i < n; ++i)
assert(v[i] == 0);
}
int main(int argc, char*argv[])
{
v(1000);
return 0;
}
