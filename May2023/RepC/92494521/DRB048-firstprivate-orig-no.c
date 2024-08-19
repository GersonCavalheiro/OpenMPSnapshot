void foo(int * a, int n, int g)
{
int i;
#pragma omp parallel for firstprivate (g)
for (i=0;i<n;i++)
{
a[i] = a[i]+g;
}
}
int a[100];
int main()
{
foo(a, 100, 7);
return 0;
}  
