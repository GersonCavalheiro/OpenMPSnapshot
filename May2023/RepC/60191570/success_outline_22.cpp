int b;
struct A
{
int c;
void f(void);
};
void A::f(void) 
{
int a;
#pragma hlt outline packed
{
a = 3;
b = 4;
c = 5;
}
}
