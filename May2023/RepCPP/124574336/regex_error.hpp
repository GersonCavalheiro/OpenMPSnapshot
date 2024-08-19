
#ifndef BOOST_XPRESSIVE_REGEX_ERROR_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_REGEX_ERROR_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <string>
#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <boost/current_function.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/info.hpp>
#include <boost/xpressive/regex_constants.hpp>

#ifdef BOOST_XPRESSIVE_DOXYGEN_INVOKED
namespace std
{
struct runtime_error {};
}
#endif

namespace boost { namespace xpressive
{

struct regex_error
: std::runtime_error
, boost::exception
{
explicit regex_error(regex_constants::error_type code, char const *str = "")
: std::runtime_error(str)
, boost::exception()
, code_(code)
{
}

regex_constants::error_type code() const
{
return this->code_;
}

virtual ~regex_error() throw()
{}

private:

regex_constants::error_type code_;
};

namespace detail
{
inline bool ensure_(
bool cond
, regex_constants::error_type code
, char const *msg
, char const *fun
, char const *file
, unsigned long line
)
{
if(!cond)
{
#ifndef BOOST_EXCEPTION_DISABLE
boost::throw_exception(
boost::xpressive::regex_error(code, msg)
<< boost::throw_function(fun)
<< boost::throw_file(file)
<< boost::throw_line((int)line)
);
#else
boost::throw_exception(boost::xpressive::regex_error(code, msg));
#endif
}
return true;
}
}

#define BOOST_XPR_ENSURE_(pred, code, msg)                                                          \
boost::xpressive::detail::ensure_(!!(pred), code, msg, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__)  \


}} 

#endif
