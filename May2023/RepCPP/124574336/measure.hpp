#if !defined(BOOST_SPIRIT_TEST_BENCHMARK_HPP)
#define BOOST_SPIRIT_TEST_BENCHMARK_HPP

#ifdef _MSC_VER
# pragma inline_recursion(on) 
# pragma inline_depth(255)    
# define _SECURE_SCL 0 
#endif

#include "high_resolution_timer.hpp"
#include <iostream>
#include <cstring>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

namespace test
{
int live_code;

template <class Accumulator>
void hammer(long const repeats)
{


const std::size_t number_of_accumulators = 1024;
live_code = 0; 

Accumulator a[number_of_accumulators];

for (long iteration = 0; iteration < repeats; ++iteration)
{
for (Accumulator* ap = a;  ap < a + number_of_accumulators; ++ap)
{
ap->benchmark();
}
}

for (Accumulator* ap = a; ap < a + number_of_accumulators; ++ap)
{
live_code += ap->val;
}
}

template <class Accumulator>
double measure(long const repeats)
{
hammer<Accumulator>(repeats);
hammer<Accumulator>(repeats);

util::high_resolution_timer time;
hammer<Accumulator>(repeats);   
return time.elapsed();          
}

template <class Accumulator>
void report(char const* name, long const repeats)
{
std::cout.precision(10);
std::cout << name << ": ";
for (int i = 0; i < (20-int(strlen(name))); ++i)
std::cout << ' ';
std::cout << std::fixed << test::measure<Accumulator>(repeats) << " [s] ";
Accumulator acc; 
acc.benchmark(); 
std::cout << std::hex << "{checksum: " << acc.val << "}";
std::cout << std::flush << std::endl;
}

struct base
{
base() : val(0) {}
int val;    
};

#define BOOST_SPIRIT_TEST_HAMMER(r, data, elem)                     \
test::hammer<elem>(repeats);


#define BOOST_SPIRIT_TEST_MEASURE(r, data, elem)                    \
test::report<elem>(BOOST_PP_STRINGIZE(elem), repeats);          \


#define BOOST_SPIRIT_TEST_BENCHMARK(max_repeats, FSeq)              \
long repeats = 100;                                             \
double measured = 0;                                            \
while (measured < 2.0 && repeats <= max_repeats)                \
{                                                               \
repeats *= 10;                                              \
util::high_resolution_timer time;                           \
BOOST_PP_SEQ_FOR_EACH(BOOST_SPIRIT_TEST_HAMMER, _, FSeq)    \
measured = time.elapsed();                                  \
}                                                               \
BOOST_PP_SEQ_FOR_EACH(BOOST_SPIRIT_TEST_MEASURE, _, FSeq)       \

}

#endif
