#include<cassert>
class Object
{
int _x;
public:
Object() : _x(42) {}
Object(int x) : _x(x) {}
Object(const Object& o) : _x(o._x) {}
~Object() {}
int get_x() const { return _x; }
};
struct A
{
int foo(const Object& o)
{
int error;
#pragma omp task shared(error)
error = (o.get_x() != 13);
#pragma omp taskwait
return error;
}
};
template < class E>
struct B
{
int foo(const E& e)
{
int error;
#pragma omp task shared(error)
error = (e.get_x() != 13);
#pragma omp taskwait
return error;
}
};
int main()
{
Object o(13);
A a;
assert(a.foo(o) == 0);
B<Object> b;
assert(b.foo(o) == 0);
}
