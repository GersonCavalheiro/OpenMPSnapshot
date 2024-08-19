#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <cstdlib>
using namespace std;
void test_isnan(){
float in0 { 0.42 };
bool out1_host {};
bool out1_device {};
{
out1_host = isnan(in0);
}
#pragma omp target map(from: out1_device)
{
out1_device = isnan(in0);
}
{
if ( out1_host != out1_device ) {
std::cerr << std::setprecision (std::numeric_limits<bool>::max_digits10 )
<< "Host: " << out1_host << " GPU: " << out1_device << std::endl;
std::exit(112);
}
}
}
int main()
{
test_isnan();
}
