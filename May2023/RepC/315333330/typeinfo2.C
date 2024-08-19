#pragma GCC visibility push (hidden)
struct A
{
A();
virtual ~A() { }
};
A::A()
{
}
void foo(A *a)
{
delete a;
}
