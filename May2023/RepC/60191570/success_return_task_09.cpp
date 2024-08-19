#include<assert.h>
class A
{
public:
int bar()
{
int x = foo() + foo();
#pragma omp taskwait on(x)
return x;
}
private:
#pragma omp task
int foo() { return 1; }
};
int main()
{
A a;
int x = a.bar();
assert(x == 2);
}
