
#include <primesum-internal.hpp>
#include <primesum.hpp>
#include <int128_t.hpp>
#include <int256_t.hpp>

#include <stdint.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <exception>
#include <sstream>
#include <ctime>

#define CHECK_11(f1, f2) check_equal(#f1, x, f1 (x), f2 (x))

#define CHECK_21(f1, f2) check_equal(#f1, x, f1 (x, get_num_threads()), f2 (x))

#define CHECK_22(f1, f2) check_equal(#f1, x, f1 (x, get_num_threads()), f2 (x, get_num_threads()))

#define CHECK_EQUAL(f1, f2, check, iters) \
{ \
cout << "Testing " << #f1 << "(x)" << flush; \
\
\
for (int64_t x = 0; x < 10000; x++) \
check(f1, f2); \
\
int64_t x = 0; \
\
for (int64_t i = 0; i < iters; i++, x += get_rand()) \
{ \
check(f1, f2); \
double percent = 100.0 * (i + 1.0) / iters; \
cout << "\rTesting " << #f1 "(x) " << (int) percent << "%" << flush; \
} \
\
cout << endl; \
}

using namespace std;
using namespace primesum;

namespace {

int get_rand()
{
return (rand() % 10000) * 1000 + 1;
}

void check_equal(const string& f1, int64_t x, int256_t res1, int256_t res2)
{
if (res1 != res2)
{
ostringstream oss;
oss << f1 << "(" << x << ") = " << res1
<< " is an error, the correct result is " << res2;
throw primesum_error(oss.str());
}
}

#ifdef _OPENMP

void test_phi_thread_safety(int64_t iters)
{
cout << "Testing phi(x, a)" << flush;

int64_t sum1 = 0;
int64_t sum2 = 0;

for (int64_t i = 0; i < iters; i++)
sum1 += pi_legendre(10000000 + i, 1);

#pragma omp parallel for reduction(+: sum2)
for (int64_t i = 0; i < iters; i++)
sum2 += pi_legendre(10000000 + i, 1);

if (sum1 != sum2)
throw primesum_error("Error: multi-threaded phi(x, a) is broken.");

std::cout << "\rTesting phi(x, a) 100%" << endl;
}

#endif

} 

namespace primesum {

bool test()
{
set_print(false); 
srand(static_cast<unsigned>(time(0)));

try
{
#ifdef _OPENMP
test_phi_thread_safety(100);
#endif

CHECK_EQUAL(pi_lmo1,                         prime_sum_tiny,      CHECK_11,  50);
CHECK_EQUAL(pi_lmo2,                         pi_lmo1,             CHECK_11,  100);
CHECK_EQUAL(pi_lmo3,                         pi_lmo2,             CHECK_11,  300);
CHECK_EQUAL(pi_lmo4,                         pi_lmo3,             CHECK_11,  400);
CHECK_EQUAL(pi_lmo5,                         pi_lmo4,             CHECK_11,  600);
CHECK_EQUAL(pi_lmo_parallel1,                pi_lmo5,             CHECK_21,  600);
CHECK_EQUAL(pi_deleglise_rivat_parallel1,    pi_lmo_parallel1,    CHECK_22,  900);
}
catch (exception& e)
{
cerr << endl << e.what() << endl;
return false;
}

cout << "All tests passed successfully!" << endl;
return true;
}

} 
