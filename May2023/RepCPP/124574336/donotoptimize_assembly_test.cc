#include <benchmark/benchmark.h>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type"
#endif

extern "C" {

extern int ExternInt;
extern int ExternInt2;
extern int ExternInt3;

inline int Add42(int x) { return x + 42; }

struct NotTriviallyCopyable {
NotTriviallyCopyable();
explicit NotTriviallyCopyable(int x) : value(x) {}
NotTriviallyCopyable(NotTriviallyCopyable const&);
int value;
};

struct Large {
int value;
int data[2];
};

}
extern "C" void test_with_rvalue() {
benchmark::DoNotOptimize(Add42(0));
}

extern "C" void test_with_large_rvalue() {
benchmark::DoNotOptimize(Large{ExternInt, {ExternInt, ExternInt}});
}

extern "C" void test_with_non_trivial_rvalue() {
benchmark::DoNotOptimize(NotTriviallyCopyable(ExternInt));
}

extern "C" void test_with_lvalue() {
int x = 101;
benchmark::DoNotOptimize(x);
}

extern "C" void test_with_large_lvalue() {
Large L{ExternInt, {ExternInt, ExternInt}};
benchmark::DoNotOptimize(L);
}

extern "C" void test_with_non_trivial_lvalue() {
NotTriviallyCopyable NTC(ExternInt);
benchmark::DoNotOptimize(NTC);
}

extern "C" void test_with_const_lvalue() {
const int x = 123;
benchmark::DoNotOptimize(x);
}

extern "C" void test_with_large_const_lvalue() {
const Large L{ExternInt, {ExternInt, ExternInt}};
benchmark::DoNotOptimize(L);
}

extern "C" void test_with_non_trivial_const_lvalue() {
const NotTriviallyCopyable Obj(ExternInt);
benchmark::DoNotOptimize(Obj);
}

extern "C" int test_div_by_two(int input) {
int divisor = 2;
benchmark::DoNotOptimize(divisor);
return input / divisor;
}

extern "C" int test_inc_integer() {
int x = 0;
for (int i=0; i < 5; ++i)
benchmark::DoNotOptimize(++x);
return x;
}

extern "C" void test_pointer_rvalue() {
int x = 42;
benchmark::DoNotOptimize(&x);
}

extern "C" void test_pointer_const_lvalue() {
int x = 42;
int * const xp = &x;
benchmark::DoNotOptimize(xp);
}

extern "C" void test_pointer_lvalue() {
int x = 42;
int *xp = &x;
benchmark::DoNotOptimize(xp);
}
