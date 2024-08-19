struct A
{
void f(int n) const
{
if (n > 0)
{
#pragma omp task
{
this->f(n-1);
}
#pragma omp taskwait
}
}
#pragma omp task
void g(int n) const
{
}
};
int main()
{
A a;
a.f(2);
a.g(2);
#pragma omp taskwait
}
