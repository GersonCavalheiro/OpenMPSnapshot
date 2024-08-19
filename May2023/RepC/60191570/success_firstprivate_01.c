#include<assert.h>
int main(int argc, char* argv[])
{
int v[10];
for(int i = 0; i < 10; ++i)
v[i] = 0;
#pragma omp task firstprivate(v)
{
for(int i = 0; i < 10; ++i)
v[i]++;
}
#pragma omp taskwait
for(int i = 0; i < 10; ++i)
assert( v[i] == 0);
return 0;
}
