typedef __SIZE_TYPE__ size_t;
extern void *acc_copyin (void *, size_t);
#define N 2
void
foo (void)
{
int a[N];
int *p = (int *)acc_copyin (&a[0], sizeof (a));
#pragma acc kernels deviceptr (p) pcopy(a)
{
a[0] = 0;
*p = 1;
}
}
