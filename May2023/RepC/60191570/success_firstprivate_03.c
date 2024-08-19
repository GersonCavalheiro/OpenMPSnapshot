#include<assert.h>
int v(int n)
{
int v1[n];
int v2[n];
for (int i = 0; i < n; ++i)
{
v1[i] = 0;
v2[i] = 0;
}
#pragma omp task firstprivate(v1, v2)
{
for (int i = 0; i < n; ++i)
{
v1[i]++;
v2[i]++;
}
}
#pragma omp taskwait
for (int i = 0; i < n; ++i)
{
assert(v1[i] == 0);
assert(v2[i] == 0);
}
}
int main(int argc, char*argv[])
{
v(1000);
return 0;
}
