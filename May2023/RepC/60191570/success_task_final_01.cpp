#include<assert.h>
int omp_in_final()
{
assert(0);
return -1;
}
int f(int n) {
if (n <= 0) return 0;
int res = 0;
#pragma omp task shared(res)
{
res += f(n-1);
res += omp_in_final() ? 1 : 2;
}
return res;
}
int main(int argc, char*argv[])
{
int res = 0;
#pragma omp task shared(res) final(1)
res = f(10);
#pragma omp taskwait
assert(res == 10);
}
