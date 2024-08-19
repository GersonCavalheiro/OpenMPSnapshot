#ifdef __FLOAT128__
#error "-mno-float128 should disable initially defining __FLOAT128__"
#endif
#pragma GCC target("float128")
#ifndef __FLOAT128__
#error "#pragma GCC target(\"float128\") should enable -mfloat128"
#endif
__float128
qabs (__float128 a)
{
return __builtin_fabsf128 (a);
}
