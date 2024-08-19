template <int t_arg>
struct A;
template <int t_arg>
void bar(A<t_arg> &a)
{
#pragma oss task firstprivate(a)
{
}
}
template <int t_arg>
struct A
{
static constexpr int size = t_arg;
int member[size];
void foo()
{
bar(*this);
}
};
template <>
void bar<1>(A<1> &a)
{
}
int main()
{
A<5> a;
a.foo();
}
