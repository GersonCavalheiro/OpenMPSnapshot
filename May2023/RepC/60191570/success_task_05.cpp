struct A
{
double x[10];
#pragma omp task out(x[i])
void f(int i)
{
}
};
int main()
{
}
