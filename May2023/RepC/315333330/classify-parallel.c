#define N 1024
extern unsigned int *__restrict a;
extern unsigned int *__restrict b;
extern unsigned int *__restrict c;
void PARALLEL ()
{
#pragma acc parallel loop copyin (a[0:N], b[0:N]) copyout (c[0:N])
for (unsigned int i = 0; i < N; i++)
c[i] = a[i] + b[i];
}
