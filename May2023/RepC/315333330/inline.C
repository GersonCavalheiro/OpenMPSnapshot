#pragma implementation "C.h"
#line 1 "A.h"
#pragma interface
template <class T> class A {};
#line 1 "C.h"
#pragma interface
template <class T> class C
{
public:
C() { A<T> *ap; }
~C() { }
};
#line 18 "inline.C"
int main()
{
C<int> c;
}
