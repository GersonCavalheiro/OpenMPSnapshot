extern int i;
#pragma omp threadprivate (i)
int main()
{
return i - 42;
}
