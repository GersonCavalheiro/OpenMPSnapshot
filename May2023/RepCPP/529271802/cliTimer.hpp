
#pragma once

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 5 && __GNUC_MINOR__ < 8
#define _GLIBCXX_USE_NANOSLEEP
#endif

#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <utility>

namespace CLI {

class cliTimer {
protected:
using clock = std::chrono::steady_clock;

using time_point = std::chrono::time_point<clock>;

using time_print_t = std::function<std::string(std::string, std::string)>;

std::string title_;

time_print_t time_print_;

time_point start_;

std::size_t cycles{1};

public:
static std::string Simple(std::string title, std::string time) { return title + ": " + time; }

static std::string Big(std::string title, std::string time) {
return std::string("-----------------------------------------\n") + "| " + title + " | Time = " + time + "\n" +
"-----------------------------------------";
}

public:
explicit cliTimer(std::string title = "cliTimer", time_print_t time_print = Simple)
: title_(std::move(title)), time_print_(std::move(time_print)), start_(clock::now()) {}

std::string time_it(std::function<void()> f, double target_time = 1) {
time_point start = start_;
double total_time;

start_ = clock::now();
std::size_t n = 0;
do {
f();
std::chrono::duration<double> elapsed = clock::now() - start_;
total_time = elapsed.count();
} while(n++ < 100u && total_time < target_time);

std::string out = make_time_str(total_time / static_cast<double>(n)) + " for " + std::to_string(n) + " tries";
start_ = start;
return out;
}

std::string make_time_str() const {
time_point stop = clock::now();
std::chrono::duration<double> elapsed = stop - start_;
double time = elapsed.count() / static_cast<double>(cycles);
return make_time_str(time);
}

std::string make_time_str(double time) const {
auto print_it = [](double x, std::string unit) {
const unsigned int buffer_length = 50;
std::array<char, buffer_length> buffer;
std::snprintf(buffer.data(), buffer_length, "%.5g", x);
return buffer.data() + std::string(" ") + unit;
};

if(time < .000001)
return print_it(time * 1000000000, "ns");
else if(time < .001)
return print_it(time * 1000000, "us");
else if(time < 1)
return print_it(time * 1000, "ms");
else
return print_it(time, "s");
}

std::string to_string() const { return time_print_(title_, make_time_str()); }

cliTimer &operator/(std::size_t val) {
cycles = val;
return *this;
}
};

class AutoTimer : public cliTimer {
public:
explicit AutoTimer(std::string title = "cliTimer", time_print_t time_print = Simple) : cliTimer(title, time_print) {}

~AutoTimer() { std::cout << to_string() << std::endl; }
};

}  

inline std::ostream &operator<<(std::ostream &in, const CLI::cliTimer &timer) { return in << timer.to_string(); }
