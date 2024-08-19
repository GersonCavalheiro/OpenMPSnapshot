int main()
{
int x = 0;
#pragma omp task inout(x)
{
x++;
#pragma omp taskwait inout(x)
}
#pragma omp taskwait
}
