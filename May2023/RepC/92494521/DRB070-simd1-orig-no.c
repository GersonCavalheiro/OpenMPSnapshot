int a[100], b[100], c[100];
int main()
{
int i;
#pragma omp simd
for (i=0;i<100;i++)
a[i]=b[i]*c[i];
return 0;
}
