#include "omp.h"
class A {
public :
A & operator+= ( const A & )
{
}
};
class B : public A {
};
#pragma omp declare reduction( + : A : omp_out += omp_in )
int main (int argc, char* argv[])
{
A a;
B b;
#pragma omp parallel reduction ( + : b )
b;
return 0;
}
