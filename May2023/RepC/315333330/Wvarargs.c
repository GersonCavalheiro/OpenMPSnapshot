#include <stdarg.h>
void
err (int a)
{
va_list vp;
va_start (vp, a); 
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvarargs"
void
foo0 (int a, int b, ...)
{
va_list vp;
va_start (vp, a);
va_end (vp);
}
void
foo1 (int a, register int b, ...)	
{
va_list vp;
va_start (vp, b);
va_end (vp);
}
#pragma GCC diagnostic pop
void
foo2 (int a, int b, ...)
{
va_list vp;
va_start (vp, a); 
va_end (vp);
}
void
foo3 (int a, register int b, ...)	
{
va_list vp;
va_start (vp, b); 
va_end (vp);
}
