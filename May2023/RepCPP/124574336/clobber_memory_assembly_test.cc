#include <benchmark/benchmark.h>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type"
#endif

extern "C" {

extern int ExternInt;
extern int ExternInt2;
extern int ExternInt3;

}

extern "C" void test_basic() {
int x;
benchmark::DoNotOptimize(&x);
x = 101;
benchmark::ClobberMemory();
}

extern "C" void test_redundant_store() {
ExternInt = 3;
benchmark::ClobberMemory();
ExternInt = 51;
}

extern "C" void test_redundant_read() {
int x;
benchmark::DoNotOptimize(&x);
x = ExternInt;
benchmark::ClobberMemory();
x = ExternInt2;
}

extern "C" void test_redundant_read2() {
int x;
benchmark::DoNotOptimize(&x);
x = ExternInt;
benchmark::ClobberMemory();
x = ExternInt2;
benchmark::ClobberMemory();
}
