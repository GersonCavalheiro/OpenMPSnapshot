double
foo (double a)
{
return __builtin_sqrt (a);
}
#pragma GCC target "no-powerpc-gpopt"
