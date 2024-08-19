void f1 ()
{
int f1_a = 2;
float f1_b[2];
#pragma acc kernels default (present)
{
f1_b[0] = f1_a;
}
#pragma acc parallel default (present)
{
f1_b[0] = f1_a;
}
}
