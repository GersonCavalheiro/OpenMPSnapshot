extern void exit (int);
#ifdef __FMA4__
#warning "__FMA4__ should not be defined before #pragma GCC target."
#endif
#pragma GCC push_options
#pragma GCC target ("fma4")
#ifndef __FMA4__
#warning "__FMA4__ should have be defined after #pragma GCC target."
#endif
float
flt_mul_add (float a, float b, float c)
{
return (a * b) + c;
}
#pragma GCC pop_options
#ifdef __FMA4__
#warning "__FMA4__ should not be defined after #pragma GCC pop target."
#endif
double
dbl_mul_add (double a, double b, double c)
{
return (a * b) + c;
}
