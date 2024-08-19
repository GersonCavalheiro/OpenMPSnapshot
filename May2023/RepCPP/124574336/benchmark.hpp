
#pragma once

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>

namespace boost { namespace numeric { namespace ublas { namespace benchmark {

class benchmark
{
using clock = std::chrono::system_clock;
public:
benchmark(std::string const &name) : name_(name) {}
void print_header()
{
std::cout << "# benchmark : " << name_ << '\n'
<< "# size \ttime (ms)" << std::endl;
}
virtual void setup(long) {}
virtual void operation(long) {}
virtual void teardown() {}

void run(std::vector<long> const &sizes, unsigned times = 10)
{
print_header();
for (auto s : sizes)
{
setup(s);
auto start = clock::now();
for (unsigned i = 0; i != times; ++i)
operation(s);
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start);
teardown();
std::cout << s << '\t' << duration.count()*1./times << std::endl;
}
}
private:
std::string name_;
};

}}}}
