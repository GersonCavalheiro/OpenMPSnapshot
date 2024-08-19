#include "omp.h"
enum A
{
L=1,
D=2
};
A& operator+=(A& _out, A& _in)
{
return  _out = A(_out + _in);
}
#pragma omp declare reduction(+:A:omp_out += omp_in)
int main (int argc, char* argv[])
{
enum A a;
#pragma omp parallel reduction(+: a)
a;
return 0;
}
