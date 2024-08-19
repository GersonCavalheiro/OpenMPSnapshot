#define N 1024
extern unsigned int *__restrict a;
extern unsigned int *__restrict b;
extern unsigned int *__restrict c;
#pragma acc declare copyin (a, b) create (c)
#pragma acc routine worker
void ROUTINE ()
{
#pragma acc loop
for (unsigned int i = 0; i < N; i++)
c[i] = a[i] + b[i];
}
