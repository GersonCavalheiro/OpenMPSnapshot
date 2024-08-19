#ifndef DATE_TIME_C_TIME_HPP___
#define DATE_TIME_C_TIME_HPP___






#include <ctime>
#include <string> 
#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <boost/date_time/compiler_config.hpp>

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; using ::time; using ::localtime;
using ::tm;  using ::gmtime; }
#endif 

#ifdef BOOST_HAS_GETTIMEOFDAY
#include <sys/time.h>
#endif

#ifdef BOOST_HAS_FTIME
#include <time.h>
#endif

namespace boost {
namespace date_time {

struct c_time {
public:
#if defined(BOOST_DATE_TIME_HAS_REENTRANT_STD_FUNCTIONS)
inline
static std::tm* localtime(const std::time_t* t, std::tm* result)
{
#if defined(__VMS) && __INITIAL_POINTER_SIZE == 64
std::tm tmp;
if(!localtime_r(t,&tmp))
result = 0;
else
*result = tmp;
#else
result = localtime_r(t, result);
#endif
if (!result)
boost::throw_exception(std::runtime_error("could not convert calendar time to local time"));
return result;
}
inline
static std::tm* gmtime(const std::time_t* t, std::tm* result)
{
#if defined(__VMS) && __INITIAL_POINTER_SIZE == 64
std::tm tmp;
if(!gmtime_r(t,&tmp))
result = 0;
else
*result = tmp;
#else
result = gmtime_r(t, result);
#endif
if (!result)
boost::throw_exception(std::runtime_error("could not convert calendar time to UTC time"));
return result;
}
#else 

#if defined(__clang__) 
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif (defined(_MSC_VER) && (_MSC_VER >= 1400))
#pragma warning(push) 
#pragma warning(disable : 4996) 
#endif
inline
static std::tm* localtime(const std::time_t* t, std::tm* result)
{
result = std::localtime(t);
if (!result)
boost::throw_exception(std::runtime_error("could not convert calendar time to local time"));
return result;
}
inline
static std::tm* gmtime(const std::time_t* t, std::tm* result)
{
result = std::gmtime(t);
if (!result)
boost::throw_exception(std::runtime_error("could not convert calendar time to UTC time"));
return result;
}
#if defined(__clang__) 
#pragma clang diagnostic pop
#elif (defined(_MSC_VER) && (_MSC_VER >= 1400))
#pragma warning(pop) 
#endif

#endif 
};
}} 

#endif 
