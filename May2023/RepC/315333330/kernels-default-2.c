#define N 2
void
foo (void)
{
unsigned int a[N];
#pragma acc kernels
{
a[0]++;
}
}
