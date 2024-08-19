#pragma GCC push_options
#pragma GCC target "arch=armv8-a+nofp"
static void
fn1 ()
{
}
#pragma GCC pop_options
float
fn2 (float a)
{
return a + 2.0;
}
