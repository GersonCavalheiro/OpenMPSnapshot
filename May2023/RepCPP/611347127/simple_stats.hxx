#pragma once

#include <cstdio> 
#include <algorithm> 
#include <cmath> 
#include <string> 

#include "status.hxx" 

namespace simple_stats {

template <typename real_t=double>
class Stats {
public:

Stats(int const value=0) { set(); } 


void add(real_t const x, real_t const weight=1) {
auto const w8 = std::abs(weight);
v[0] += w8;
v[1] += w8*x;
v[2] += w8*x*x;
maxi = std::max(maxi, x);
mini = std::min(mini, x);
++times;
} 


void get(double values[8]) const { 
values[0] = v[0];
values[1] = v[1];
values[2] = v[2];
values[3] = 0; 
values[4] = times;
values[5] = 0; 
values[6] = -mini; 
values[7] =  maxi; 
} 

void set(double const values[8]=nullptr) { 
if (nullptr != values) {
times = values[4];
mini = -values[6];
maxi =  values[7];
v[0] =  values[0];
v[1] =  values[1];
v[2] =  values[2];
} else { 
times = 0;
float constexpr LIM = 1.7e38;
mini =  LIM;
maxi = -LIM;
for (int p = 0; p < 3; ++p) v[p] = 0;
} 
} 

real_t min() const { return mini*(times > 0); }
real_t max() const { return maxi*(times > 0); }
real_t num() const { return v[0]; }
real_t sum() const { return v[1]; }
size_t tim() const { return times; }
double mean() const { return (v[0] > 0) ? v[1]/double(v[0]) : 0.0; }
double variance() const {
auto const mu = mean();
return (times > 0 && v[0] > 0) ?
std::max(0.0, v[2]/v[0] - mu*mu) : 0.0;
} 
double dev() const { return std::sqrt(variance()); } 

std::string interval(double const f=1) const {
char buffer[96]; std::snprintf(buffer, 96, "[%g, %g +/- %g, %g]", min()*f, mean()*f, dev()*f, max()*f);
return std::string(buffer);
} 

private:
size_t times;
real_t mini, maxi;
real_t v[3];
}; 





#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename real_t>
inline status_t test_basic(int const echo=0, int offset=0, double const threshold=1e-6) {
Stats<real_t> s;
int const begin=offset, end=offset + 100;
double const ref[] = {49.5 + offset, 833.25}; 
for (int i = begin; i < end; ++i) {
s.add(i);
} 
auto const mean = s.mean();
if (echo > 3) std::printf("# %s: from %d to %d: %g +/- %g\n", __func__, begin, end - 1, mean, s.dev());
auto const dev_mean = std::abs(ref[0] - mean),
dev_variance = std::abs(ref[1] - s.variance());
if (echo > 7) std::printf("# %s: dev_mean= %g, dev_variance= %g\n", __func__, dev_mean, dev_variance);
if (echo > 9) std::printf("# %s: %ld Byte\n", __FILE__, sizeof(s));
return (dev_mean > threshold*mean) + (dev_variance > threshold*mean*mean);
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("\n# %s %s\n", __FILE__, __func__);
status_t stat(0);
stat += test_basic<float >(echo);
stat += test_basic<double>(echo);
stat += test_basic<float >(echo, 1000000);
stat += test_basic<double>(echo, 1000000);
return stat;
} 

#endif 

} 
