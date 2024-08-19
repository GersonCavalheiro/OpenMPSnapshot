



#ifndef BOOST_RE_PAT_EXCEPT_HPP
#define BOOST_RE_PAT_EXCEPT_HPP

#ifndef BOOST_REGEX_CONFIG_HPP
#include <boost/regex/config.hpp>
#endif

#include <stdexcept>
#include <cstddef>
#include <boost/regex/v4/error_type.hpp>

namespace boost{

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 4275)
#if BOOST_MSVC >= 1800
#pragma warning(disable : 26812)
#endif
#endif
class BOOST_REGEX_DECL regex_error : public std::runtime_error
{
public:
explicit regex_error(const std::string& s, regex_constants::error_type err = regex_constants::error_unknown, std::ptrdiff_t pos = 0);
explicit regex_error(regex_constants::error_type err);
~regex_error() BOOST_NOEXCEPT_OR_NOTHROW;
regex_constants::error_type code()const
{ return m_error_code; }
std::ptrdiff_t position()const
{ return m_position; }
void raise()const;
private:
regex_constants::error_type m_error_code;
std::ptrdiff_t m_position;
};

typedef regex_error bad_pattern;
typedef regex_error bad_expression;

namespace BOOST_REGEX_DETAIL_NS{

BOOST_REGEX_DECL void BOOST_REGEX_CALL raise_runtime_error(const std::runtime_error& ex);

template <class traits>
void raise_error(const traits& t, regex_constants::error_type code)
{
(void)t;  
std::runtime_error e(t.error_string(code));
::boost::BOOST_REGEX_DETAIL_NS::raise_runtime_error(e);
}

}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

} 

#endif



