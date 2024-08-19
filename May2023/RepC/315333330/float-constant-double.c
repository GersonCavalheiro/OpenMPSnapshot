#include <float.h>
extern double a, b, c, d;
void
foo ()
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 ON")
a = 0.1d * DBL_MAX;
b = DBL_EPSILON * 10.0d;
c = DBL_MIN * 200.0d;
}
