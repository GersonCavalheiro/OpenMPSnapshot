#include <stdio.h>
#include <stdlib.h>
#define N 100
class myInt {
int x;
public:
myInt() : x(0) {}
#pragma omp declare reduction( + : myInt : omp_out += omp_in)
myInt & operator+= (const myInt &b) {  this->x += b.x; return *this; }
myInt & operator+= (const int b) { this->x += b; return *this; }
myInt & operator-= (const myInt &b) {  this->x -= b.x; return *this; }
myInt & operator-= (const int b) { this->x -= b; return *this; }
myInt & operator*= (const myInt &b) {  this->x *= b.x; return *this; }
myInt & operator*= (const int b) { this->x *= b; return *this; }
int getX() const 
{ 
return x; 
}
void foo();
int bar()
{
int x;
#pragma omp declare reduction( * : myInt : omp_out *= omp_in)
myInt a;
#pragma omp parallel reduction(* : a)
x = rand();
return x;
}
};
void myInt::foo()
{
int x;
#pragma omp declare reduction(- : myInt : omp_out -= omp_in)
myInt a;
#pragma omp parallel reduction(- :a)
x = rand();
}
int main (int argc, char **argv)
{
int i,s=0;
int a[N];
myInt x;
for ( i = 0; i < N ; i++ ) {
a[i] = i;
s += i;
}
#pragma omp parallel for reduction(myInt::operator+ :x)
for ( i = 0; i < N ; i++ )
{
x += a[i];
}
if ( x.getX() != s ) abort();
return 0;
}
