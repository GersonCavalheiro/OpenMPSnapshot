#include <complex>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <cstdlib>
using namespace std;
bool almost_equal(complex<double> x, complex<double> y, int ulp) {
return std::abs(x-y) <= std::numeric_limits<double>::epsilon() * std::abs(x+y) * ulp || std::abs(x-y) < std::numeric_limits<double>::min();
}
void test_asinh(){
const char* usr_precision = getenv("OVO_TOL_ULP");
const int precision = usr_precision ? atoi(usr_precision) : 4;
complex<double> in0 { 0.42, 0.0 };
complex<double> out1_device {};
#pragma omp target map(from: out1_device)
{
out1_device = asinh(in0);
}
{
if ( !almost_equal(sinh(out1_device), in0, 2*precision) ) {
std::cerr << std::setprecision (std::numeric_limits<double>::max_digits10 )
<< "Expected:" << in0 << " Got: " << sinh(out1_device) << std::endl;
std::exit(112);
}
}
}
int main()
{
test_asinh();
}
