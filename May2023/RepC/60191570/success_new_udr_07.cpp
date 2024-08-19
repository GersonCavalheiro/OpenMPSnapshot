#include <iostream>
#include <stdlib.h>
#include "omp.h"
#define N 100
namespace A {
struct myInt {
int x;
myInt() : x(0) { }
void operator+= ( const int b ) { x += b; }
void operator+= ( const myInt &b ) { x += b.x; }
};
#pragma omp declare reduction(plus:myInt: omp_out.x += omp_in.x)
}
#pragma omp declare reduction(plus:A::myInt: omp_out.x = omp_in.x)
int main (int argc, char **argv)
{
int i,s=0;
int a[N];
A::myInt x;
for ( i = 0; i < N ; i++ ) {
a[i] = i;
s += i;
}
#pragma omp parallel for reduction(A::plus:x)
for ( i = 0; i < N ; i++ )
{
x += a[i];
}
if ( x.x != s ) 
{
std::cerr << "(x.x == " << x.x << ") != (s == " << s << ")" << std::endl;
abort();
}
return 0;
}
