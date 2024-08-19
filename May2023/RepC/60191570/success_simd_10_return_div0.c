#pragma omp simd linear(n) 
int foo(int n)
{
int result;
if (n > 2)
{
if (n < 100)
{
if (n > 50)
return 1;
return 2;
}
else
{
result = n / 0;
}
result = n / 0;
}
return result;
}
int main()
{
int i;
#pragma omp simd
for (i=0; i<100; i++)
{
foo(i);
}
}
