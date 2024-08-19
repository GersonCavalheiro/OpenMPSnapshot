typedef __SIZE_TYPE__ size_t;
extern void *acc_copyin (void *, size_t);
void
foo (void)
{
int a = 0;
int *p = (int *)acc_copyin (&a, sizeof (a));
#pragma acc kernels deviceptr (p) pcopy(a)
{
a = 0;
*p = 1;
}
}
