#include<unistd.h>
#include<assert.h>
int main()
{
int x = -1;
int *p1 = &x;
int *p2=  &(*p1);
#pragma omp task inout(*p1)
{
sleep(1);
*p1 = 2;
}
#pragma omp task inout(*p2)
{
assert(*p2 == 2);
}
#pragma omp taskwait
return 0;
}
