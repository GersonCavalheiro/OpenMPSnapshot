
#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/gamma.hpp>
#include "bindings.hpp"
#include "../../test/test_gamma.hpp"

BOOST_AUTO_TEST_CASE_EXPECTED_FAILURES(test_main, 10000);

BOOST_AUTO_TEST_CASE(test_main)
{
BOOST_MATH_CONTROL_FP;

error_stream_replacer rep;

#ifdef TYPE_TO_TEST

test_gamma(static_cast<TYPE_TO_TEST>(0), NAME_OF_TYPE_TO_TEST);

#else
bool test_float = false;
bool test_double = false;
bool test_long_double = false;

if(std::numeric_limits<long double>::digits == std::numeric_limits<double>::digits)
{
if(BOOST_MATH_PROMOTE_FLOAT_POLICY == false)
test_float = true;
test_double = true;
}
else
{
if(BOOST_MATH_PROMOTE_FLOAT_POLICY == false)
test_float = true;
if(BOOST_MATH_PROMOTE_DOUBLE_POLICY == false)
test_double = true;
test_long_double = true;
}

#ifdef ALWAYS_TEST_DOUBLE
test_double = true;
#endif

if(test_float)
test_gamma(0.0f, "float");
if(test_double)
test_gamma(0.0, "double");
if(test_long_double)
test_gamma(0.0L, "long double");
#ifdef BOOST_MATH_USE_FLOAT128
#endif


#endif
}

