#include <stdlib.h>
int foo(int p)
{
#pragma analysis_check assert upper_exposed(p)
return p;
}
int bar(int p)
{
#pragma analysis_check assert upper_exposed(p)
return p;
}
int function(int (*fun)(int), int p)
{
#pragma analysis_check assert undefined(p)
return (*fun)(p);
}
int main(int argc, char *argv[])
{
int a, b;
#pragma analysis_check assert upper_exposed(argc) defined(argc, a)
a = function((int (*)(int))foo, argc++);
#pragma analysis_check assert upper_exposed(argc) defined(argc, b)
b = function((int (*)(int))bar, argc++);
#pragma analysis_check assert upper_exposed(a, b)
if( a != 1 || b != 2 )
exit( 1 );
return 0;
}
