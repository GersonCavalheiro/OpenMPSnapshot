#include<unistd.h>
#include<assert.h>
int main()
{
int x[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
int *p = &(x[1]);
#pragma omp task out(x[1])
{
sleep(1);
x[1] = 2;
}
#pragma omp task in([1]p)
{
assert(*p == 2);
}
#pragma omp taskwait
return 0;
}
