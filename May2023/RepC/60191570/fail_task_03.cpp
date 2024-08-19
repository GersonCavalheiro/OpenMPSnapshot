int m;
struct B
{
int n;
};
struct A
{
int n;
B* b;
#pragma omp task inout(this->n, this->b->n)
void f();
};
