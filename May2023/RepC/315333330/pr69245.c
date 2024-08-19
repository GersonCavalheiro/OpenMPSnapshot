#pragma GCC target "fpu=vfp"
#pragma GCC push_options
#pragma GCC target "fpu=neon-vfpv4"
int a, c, d;
float b;
static int fn1 ()
{
return 0;
}
#pragma GCC pop_options
void fn2 ()
{
d = b * c + a;
}
