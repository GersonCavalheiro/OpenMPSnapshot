namespace N
{
struct A
{
void foo();
void bar()
{
int x = 0;
#pragma omp task reduction(+: x)
{
x++;
}
#pragma omp taskwait
}
};
void A::foo()
{
int x = 0;
#pragma omp task reduction(+: x)
{
x++;
}
#pragma omp taskwait
}
}
int main()
{
N::A a;
a.foo();
a.bar();
}
