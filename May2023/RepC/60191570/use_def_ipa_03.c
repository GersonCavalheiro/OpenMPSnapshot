#include <stdlib.h>
int fib( int n )
{
#pragma analysis_check assert upper_exposed(n)
if( n == 1 || n == 2)
return 1;
int x, y;
#pragma analysis_check assert defined(x) upper_exposed(n)
x = fib( n );
#pragma analysis_check assert defined(y) upper_exposed(n)
y = fib( n - 1 );
return x + y;
}
int main( int argc, char** argv )
{
int n = fib( 10 ) ;
if( n != 55 )
exit( 1 );
return 0;
}