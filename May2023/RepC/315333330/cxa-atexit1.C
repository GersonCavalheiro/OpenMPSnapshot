# 1 "cxa-atexit1.C"
struct A
{
struct B
{
B ();
~B ();
};
};
static A::B b;
# 1 "cxa-atexit1.h" 1
#pragma interface
template <class T> struct C
{
~C (void);
};
struct D : public C<bool>
{
D (void) : C<bool> () { }
};
# 55 "cxa-atexit1.C" 2
