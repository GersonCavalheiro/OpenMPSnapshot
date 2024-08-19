#include<unistd.h>
#include<assert.h>
struct A
{
int x;
};
int main()
{
struct A a = { -1 };
int *p = &a.x;
#pragma omp task out(a.x)
{
sleep(1);
a.x = 2;
}
#pragma omp task in(*p)
{
assert(*p == 2);
}
#pragma omp taskwait
return 0;
}
