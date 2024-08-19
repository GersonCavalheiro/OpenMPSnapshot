#pragma GCC visibility push(hidden)
class Foo
{
__attribute__ ((visibility ("internal"))) void method();
};
#pragma GCC visibility pop
void Foo::method() { }
