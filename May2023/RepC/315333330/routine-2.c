template <typename T>
extern T one_d();
#pragma acc routine (one_d) 
template <typename T>
T
one()
{
return 1;
}
#pragma acc routine (one) 
int incr (int);
float incr (float);
#pragma acc routine (incr) 
int sum (int, int);
namespace foo {
#pragma acc routine (sum)
int sub (int, int);
}
#pragma acc routine (foo::sub)
int decr (int a);
#pragma acc routine
float decr (float a);
