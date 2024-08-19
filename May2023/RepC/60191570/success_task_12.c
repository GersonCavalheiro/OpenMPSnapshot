#include<assert.h>
struct C
{
int z;
};
int main()
{
struct C c;
c.z = -1;
#pragma omp task inout(c.z)
{
c.z = 2;
}
#pragma omp taskwait
assert(c.z == 2);
return 0;
}
