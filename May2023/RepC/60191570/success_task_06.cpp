struct A
{
int x[10];
#pragma omp task out(x[i])
void f(int i)
{
x[i] = 0;
}
};
int main()
{
A a;
A* ptr_a = &a;
a.f(0);
ptr_a->f(1);
#pragma omp taskwait
}
