#pragma weak foo
template <typename T>
struct A { };
template <typename T>
void bar (A<T> &);
