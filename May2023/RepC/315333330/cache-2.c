int
main (int argc, char **argv)
{
#define N   2
int a[N], b[N];
int i;
for (i = 0; i < N; i++)
{
a[i] = 3;
b[i] = 0;
}
#pragma acc parallel copyin (a[0:N]) copyout (b[0:N])
{
int ii;
for (ii = 0; ii < N; ii++)
{
const int idx = ii;
int n = 1;
const int len = n;
#pragma acc cache 
#pragma acc cache a[0:N] 
#pragma acc cache (a) 
#pragma acc cache ( 
#pragma acc cache () 
#pragma acc cache (,) 
#pragma acc cache (a[0:N] 
#pragma acc cache (a[0:N],) 
#pragma acc cache (a[0:N]) copyin (a[0:N]) 
#pragma acc cache () 
#pragma acc cache (a[0:N] b[0:N]) 
#pragma acc cache (a[0:N] b[0:N}) 
#pragma acc cache (a[0:N] 
#pragma acc cache (a[0:N]) ( 
#pragma acc cache (a[0:N]) ii 
#pragma acc cache (a[0:N] ii) 
b[ii] = a[ii];
}
}
for (i = 0; i < N; i++)
{
if (a[i] != b[i])
__builtin_abort ();
}
return 0;
}
