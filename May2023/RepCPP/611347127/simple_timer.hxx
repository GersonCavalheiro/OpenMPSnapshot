#pragma once

#include <cstdio> 
#include <chrono> 
#include <string> 

#include "status.hxx" 
#ifndef NO_UNIT_TESTS
#include "simple_stats.hxx" 
#endif 

class SimpleTimer {
private:
std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
std::string file;
std::string func;
int line;
int echo;
public:

SimpleTimer(char const *sourcefile, int const sourceline=0, char const *function=nullptr, int const echo=2)
: file(sourcefile), func(function), line(sourceline), echo(echo) {
start_time = std::chrono::high_resolution_clock::now(); 
} 

double stop(int const stop_echo=0) const { 
auto const stop_time = std::chrono::high_resolution_clock::now(); 
auto const musec = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
if (stop_echo > 0) std::printf("# timer started at %s:%d %s took %.5f sec\n", file.c_str(), line, func.c_str(), 1e-6*musec);
return 1e-6*musec;
} 

~SimpleTimer() { stop(echo); } 

}; 







namespace simple_timer {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline int64_t fibonacci(int64_t const n) {
if (n < 3) return 1;
return fibonacci(n - 1) + fibonacci(n - 2);
} 

inline status_t test_basic_usage(int const echo=3) {
int64_t result;
int64_t const inp = 40, reference = 102334155;
{ 
SimpleTimer timer(__FILE__, __LINE__, "comment=fibonacci", echo);
result = fibonacci(inp);
} 
if (echo > 0) std::printf("# fibonacci(%lld) = %lld\n", inp, result);
return (reference != result);
} 

inline status_t test_stop_function(int const echo=3) {
int64_t result;
int64_t const inp = 40, reference = 102334155;
simple_stats::Stats<> s;
for (int i = 0; i < 5; ++i) {
SimpleTimer timer(__FILE__, __LINE__, "", 0);
result = fibonacci(inp);
s.add(timer.stop());
} 
auto const average_time = s.mean();
if (echo > 0) std::printf("# fibonacci(%lld) = %lld took %g +/- %.1e seconds per iteration\n",
inp, result, average_time, s.dev());
return (average_time < 0) + (reference != result);
} 

inline status_t all_tests(int const echo=0) {
if (echo > 2) std::printf("\n# %s %s\n\n", __FILE__, __func__);
status_t stat(0);
stat += test_basic_usage(echo);
stat += test_stop_function(echo);
return stat;
} 

#endif 

} 
