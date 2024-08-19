template<class T>
struct A {
void foo() {};
__attribute ((visibility ("hidden"))) void bar();
};
template<> void A<int>::foo() {}
template<> inline void A<long>::foo() {}
void f () { A<long> a; a.foo(); }
template<> __attribute ((visibility ("default"))) void A<int>::bar() {}
template<> void A<long>::bar() { }
#pragma GCC visibility push(default)
template<> void A<char>::bar() { }
#pragma GCC visibility pop
