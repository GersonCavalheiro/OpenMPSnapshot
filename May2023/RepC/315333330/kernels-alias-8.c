typedef __SIZE_TYPE__ size_t;
extern void *acc_copyin (void *, size_t);
void
foo (int *a, size_t n)
{
int *p = (int *)acc_copyin (a, n);
#pragma acc kernels deviceptr (p) pcopy(a[0:n])
{
a = 0;
*p = 1;
}
}
