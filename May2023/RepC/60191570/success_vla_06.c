#include<assert.h>
int v(int n) {
int v1[n];
for (int i = 0; i < n; ++i)
v1[i] = 0;
#pragma omp task private(v1)
{
for (int i = 0; i < n; ++i)
v1[i] = 7;
}
#pragma omp taskwait
for (int i = 0; i < n; ++i)
assert(v1[i] == 0);
}
int main(int argc, char*argv[])
{
v(100);
return 0;
}
