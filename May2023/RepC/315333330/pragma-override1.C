#pragma GCC visibility push(hidden)
class __attribute__ ((visibility ("internal"))) Foo
{
void method();
};
#pragma GCC visibility pop
void Foo::method() { }
