#pragma GCC visibility push(hidden)
template <class T> void bar(T) { }
#pragma GCC visibility pop
template void bar (long);
template<> void bar (double) { }
template __attribute ((visibility ("default"))) void bar (short);
template<> __attribute ((visibility ("default"))) void bar (float) { }
#pragma GCC visibility push(default)
template<> void bar(char) { }
template void bar(int);
#pragma GCC visibility pop
template <class T> __attribute ((visibility ("hidden"))) void foo(T) { }
template void foo (long);
template<> void foo (double) { }
template __attribute ((visibility ("default"))) void foo (short);
template<> __attribute ((visibility ("default"))) void foo (float) { }
#pragma GCC visibility push(default)
template<> void foo(char) { }
template void foo(int);
#pragma GCC visibility pop
