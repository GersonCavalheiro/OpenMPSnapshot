#pragma GCC target "no-powerpc-gpopt"
double
foo (double a)
{
return __builtin_sqrt (a);
}
