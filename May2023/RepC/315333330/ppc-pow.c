double
do_pow_0_75_default (double a)
{
return __builtin_pow (a, 0.75);	
}
double
do_pow_0_5_default (double a)
{
return __builtin_pow (a, 0.5);	
}
#pragma GCC target "no-powerpc-gpopt,no-powerpc-gfxopt"
double
do_pow_0_75_nosqrt (double a)
{
return __builtin_pow (a, 0.75);	
}
double
do_pow_0_5_nosqrt (double a)
{
return __builtin_pow (a, 0.5);	
}
#pragma GCC reset_options
