#pragma omp task inout(*var) priority(*var)
void foo(int* var)
{
var++;
}
int main()
{
int a;
foo(&a);
#pragma omp task priority(0)
{
}
#pragma omp taskwait
}
