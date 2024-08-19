#pragma omp simd linear(n)
int foo(int n)
{
if ((n%2) == 0)
{
return 35;
}
return 75;
}
int main()
{
int i;
int __attribute__((__aligned__(64))) a[100];
#pragma omp simd
for (i=0; i<100; i++)
{
a[i] = foo(i);
}
for (i=0; i<100; i++)
{
if (a[i] != foo(i))
{
return 1;
}
}
return 0;
}
