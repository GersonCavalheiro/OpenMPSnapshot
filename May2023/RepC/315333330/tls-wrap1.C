int i = 42;
#pragma omp threadprivate (i)
int main()
{
return i - 42;
}
