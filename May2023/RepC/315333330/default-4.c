void f1 ()
{
int f1_a = 2;
float f1_b[2];
#pragma acc data copyin (f1_a) copyout (f1_b)
{
#pragma acc kernels
{
f1_b[0] = f1_a;
}
#pragma acc parallel
{
f1_b[0] = f1_a;
}
}
}
void f2 ()
{
int f2_a = 2;
float f2_b[2];
#pragma acc data copyin (f2_a) copyout (f2_b)
{
#pragma acc kernels default (none)
{
f2_b[0] = f2_a;
}
#pragma acc parallel default (none)
{
f2_b[0] = f2_a;
}
}
}
void f3 ()
{
int f3_a = 2;
float f3_b[2];
#pragma acc data copyin (f3_a) copyout (f3_b)
{
#pragma acc kernels default (present)
{
f3_b[0] = f3_a;
}
#pragma acc parallel default (present)
{
f3_b[0] = f3_a;
}
}
}
