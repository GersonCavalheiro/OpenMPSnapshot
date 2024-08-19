#include <iostream>
#include <stdlib.h>
#define N 100
namespace A {
struct myInt {
int x;
void operator+= ( const int b ) { x += b; }
void operator+= ( const myInt &b ) { x += b.x; }
};
#pragma omp declare reduction(plus:myInt: omp_out.x += omp_in.x) initializer(omp_priv = {0})
}
int main (int argc, char **argv)
{
int i,s=0;
int a[N];
A::myInt x = {0};
for ( i = 0; i < N ; i++ ) {
a[i] = i;
s += i;
}
#pragma omp parallel for reduction(A::plus : x)
for ( i = 0; i < N ; i++ )
{
x += a[i];
}
if ( x.x != s ) abort();
return 0;
}
