template<typename T>
struct A
{
A() { }
A(const A&) { }
void foo() { }
};
int main()
{
A<int> a;
#pragma omp task firstprivate (a)
a.foo();
return 0;
}
