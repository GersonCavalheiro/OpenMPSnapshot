#pragma GCC target("hard-dfp")
_Decimal64 p1(_Decimal64 f, _Decimal64 g)
{
return f * g;
}
#pragma GCC reset_options
#pragma GCC target("no-hard-dfp")
_Decimal64 p0(_Decimal64 f, _Decimal64 g)
{
return f / 2;
}
#pragma GCC reset_options
__attribute__ ((target("hard-dfp")))
_Decimal64 a1(_Decimal64 f, _Decimal64 g)
{
return f + g;
}
__attribute__ ((target("no-hard-dfp")))
_Decimal64 a0(_Decimal64 f, _Decimal64 g)
{
return f - g;
}
_Decimal64 d(_Decimal64 f, _Decimal64 g)
{
return f - g;
}
