
#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/zeta.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

int main()
{
typedef double T;
#define SC_(x) static_cast<double>(x)
#include "zeta_data.ipp"
#include "zeta_neg_data.ipp"
#include "zeta_1_up_data.ipp"
#include "zeta_1_below_data.ipp"

add_data(zeta_data);
add_data(zeta_neg_data);
add_data(zeta_1_up_data);
add_data(zeta_1_below_data);

unsigned data_total = data.size();

screen_data([](const std::vector<double>& v){  return boost::math::zeta(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });

#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
screen_data([](const std::vector<double>& v){  return std::tr1::riemann_zeta(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
screen_data([](const std::vector<double>& v){  return gsl_sf_zeta(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });
#endif

unsigned data_used = data.size();
std::string function = "zeta[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
std::string function_short = "zeta";

double time = exec_timed_test([](const std::vector<double>& v){  return boost::math::zeta(v[0]);  });
std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
#if !defined(COMPILER_COMPARISON_TABLES)
if(sizeof(long double) != sizeof(double))
{
double time = exec_timed_test([](const std::vector<double>& v){  return boost::math::zeta(v[0], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
}
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::riemann_zeta(v[0]);  });
std::cout << time << std::endl;
report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_zeta(v[0]);  });
std::cout << time << std::endl;
report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif

return 0;
}

