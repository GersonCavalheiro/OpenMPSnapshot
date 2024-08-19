#pragma omp task out(a)
void foo(int a)
{
}
int main()
{
int a = 2;
foo(a);
}
