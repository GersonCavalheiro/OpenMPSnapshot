#pragma omp task
void foo() { }
#pragma omp task
int fii() { }
int main()
{
int x = 0;
#pragma omp task
{
#pragma omp task
{
x++;
}
#pragma omp taskwait
}
foo();
int h = fii();
#pragma omp taskwait
}
