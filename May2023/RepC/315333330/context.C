#pragma implementation "context.h"
#line 1 "context.h"
#pragma interface
template <class T>
struct A {
inline void foo () { }
class B { };
};
struct C : public A<int> {
void bar (C::B&);
};
#line 2 "context.C"
void C::bar (C::B& b) { }
