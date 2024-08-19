#include<assert.h>
void foo()
{
int x = 0, y = 2;
#pragma omp task depend(inout: x) shared(x)
x++;
#pragma omp task depend(inout: y) shared(y)
y--;
#pragma omp taskwait depend(in: x)
assert(x == 1);
#pragma omp taskwait
assert(x == y);
}
int main(int argc, char*argv)
{
foo();
return 0;
}
