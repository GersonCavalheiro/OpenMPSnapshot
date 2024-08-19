#pragma omp simd linear(n)
int foo(int n)
{
if (n > 2)
{
if (n < 100)
{
if (n > 50)
return 33;
return 2;
}
else
{
n = n / 0;
}
n = n / 0;
}
return n;
}
int main()
{
int i;
#pragma omp simd
for (i=0; i<101; i++)
{
foo(i);
}
}
