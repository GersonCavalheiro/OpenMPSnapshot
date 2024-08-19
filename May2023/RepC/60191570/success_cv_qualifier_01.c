#include <assert.h>
struct A
{
int x;
};
typedef struct A* AA;
void f(const AA a)
{
int i;
AA b = 0;
#pragma omp parallel private(i)
{
for (i = 0; i < 1; i++)
{
b = a;
}
}
assert(b->x == 42);
#pragma omp for
for (i = 0; i < 1; i++)
{
b = a;
}
assert(b->x == 42);
}
int main(int argc, char* argv[])
{
struct A a = { 42 };
f(&a);
}
