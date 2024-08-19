#include <benchmark/benchmark.h>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type"
#endif

extern "C" {
extern int ExternInt;
benchmark::State& GetState();
void Fn();
}

using benchmark::State;

extern "C" int test_for_auto_loop() {
State& S = GetState();
int x = 42;

for (auto _ : S) {
benchmark::DoNotOptimize(x);
}

return 101;
}

extern "C" int test_while_loop() {
State& S = GetState();
int x = 42;

while (S.KeepRunning()) {
benchmark::DoNotOptimize(x);
}



return 101;
}
