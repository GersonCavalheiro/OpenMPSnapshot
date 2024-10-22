#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <cstdlib>
using namespace std;
bool almost_equal(float x, float y, int ulp) {
return std::fabs(x-y) <= std::numeric_limits<float>::epsilon() * std::fabs(x+y) * ulp || std::fabs(x-y) < std::numeric_limits<float>::min();
}
void test_remquof(){
const char* usr_precision = getenv("OVO_TOL_ULP");
const int precision = usr_precision ? atoi(usr_precision) : 4;
float in0 { 0.42 };
float in1 { 0.42 };
int out2_host {};
int out2_device {};
float out3_host {};
float out3_device {};
{
out3_host = remquof(in0, in1, &out2_host);
}
#pragma omp target map(from: out2_device, out3_device)
{
out3_device = remquof(in0, in1, &out2_device);
}
{
if ( out2_host != out2_device ) {
std::cerr << std::setprecision (std::numeric_limits<int>::max_digits10 )
<< "Host: " << out2_host << " GPU: " << out2_device << std::endl;
std::exit(112);
}
if ( !almost_equal(out3_host,out3_device, precision) ) {
std::cerr << std::setprecision (std::numeric_limits<float>::max_digits10 )
<< "Host: " << out3_host << " GPU: " << out3_device << std::endl;
std::exit(112);
}
}
}
int main()
{
test_remquof();
}
