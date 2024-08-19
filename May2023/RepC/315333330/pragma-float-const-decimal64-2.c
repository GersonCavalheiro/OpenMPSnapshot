_Pragma ("STDC FLOAT_CONST_DECIMAL64 ON")
double a = 1.0 * 2.0dd;
double
f1 (void)
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 OFF")
double b = 2.0 * 3.0d;
{
double c = 3.0 * 4.0d;
b = b + c;
}
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 ON")
double d = 4.0 * 5.0dd;
b = b + d;
}
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 DEFAULT")
double e = 5.0 * 6.0d;
b = b + e;
}
return b;
}
double
f2 (void)
{
double b = 2.0 * 3.0dd;
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 OFF")
double c = 3.0 * 4.0d;
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 ON")
double d = 4.0 * 5.0dd;
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 DEFAULT")
double e = 5.0 * 6.0d;
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 ON")
double f = 6.0 * 7.0dd;
b = a + b + c + d + e + f;
}
}
}
}
return b;
}
double f = 6.0 * 7.0dd;
double
f3 (void)
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 OFF")
double b = 2.0 * 3.0d;
return b + f;
}
double g = 7.0 + 8.0dd;
