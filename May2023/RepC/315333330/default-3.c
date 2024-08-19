void f1 ()
{
int f1_a = 2;
float f1_b[2];
#pragma acc kernels default (none) 
{
f1_b[0] 
= f1_a; 
}
#pragma acc parallel default (none) 
{
f1_b[0] 
= f1_a; 
}
}
