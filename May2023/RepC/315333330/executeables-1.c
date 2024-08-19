void
foo (void)
{
const int N = 32;
float x[N], y[N];
int flag = 0;
if (flag)
#pragma acc update host (x[0:N]) 
flag = 1;
while (flag)
#pragma acc update host (x[0:N]) 
flag = 2;
#pragma acc enter data create (x[0:N])
{
if (flag)
#pragma acc update host (x[0:N]) 
flag = 1;
}
if (flag)
while (flag)
#pragma acc update host (x[0:N]) 
flag = 2;
if (flag)
#pragma acc wait 
flag = 1;
while (flag)
#pragma acc wait 
flag = 2;
#pragma acc enter data create (x[0:N])
{
if (flag)
#pragma acc wait 
flag = 1;
}
if (flag)
#pragma acc enter data create (x[0:N]) 
flag = 1;
while (flag)
#pragma acc enter data create (x[0:N]) 
flag = 2;
#pragma acc enter data create (x[0:N])
{
if (flag)
#pragma acc enter data create (y[0:N]) 
flag = 1;
}
if (flag)
#pragma acc exit data delete (x[0:N]) 
flag = 1;
while (flag)
#pragma acc exit data delete (x[0:N]) 
flag = 2;
#pragma acc enter data create (x[0:N])
{
if (flag)
#pragma acc exit data delete (x[0:N]) 
flag = 1;
}
}
