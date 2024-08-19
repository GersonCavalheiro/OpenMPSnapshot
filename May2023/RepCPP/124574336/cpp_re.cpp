

#define BOOST_WAVE_SOURCE 1

#include <boost/config/warning_disable.hpp>

#include <cstddef>

#include <boost/wave/wave_config.hpp>          

#if defined(BOOST_HAS_UNISTD_H)
#include <unistd.h>
#else
#include <io.h>
#endif

#include <boost/detail/workaround.hpp>

#include <boost/wave/token_ids.hpp>
#include <boost/wave/cpplexer/re2clex/scanner.hpp>
#include <boost/wave/cpplexer/re2clex/cpp_re.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

#if defined(BOOST_MSVC)
#pragma warning (disable: 4101)     
#pragma warning (disable: 4102)     
#endif

namespace boost {
namespace wave {
namespace cpplexer {
namespace re2clex {

bool is_backslash(uchar *p, uchar *end, int &len)
{
if (*p == '\\') {
len = 1;
return true;
}
else if (*p == '?' && *(p+1) == '?' && (p+2 < end && *(p+2) == '/')) {
len = 3;
return true;
}
return false;
}

uchar_wrapper::uchar_wrapper (uchar *base_cursor, std::size_t column)
:   base_cursor(base_cursor), column(column)
{}

uchar_wrapper& uchar_wrapper::operator++()
{
++base_cursor;
++column;
return *this;
}

uchar_wrapper& uchar_wrapper::operator--()
{
--base_cursor;
--column;
return *this;
}

uchar uchar_wrapper::operator* () const
{
return *base_cursor;
}

uchar_wrapper::operator uchar *() const
{
return base_cursor;
}

std::ptrdiff_t
operator- (uchar_wrapper const& lhs, uchar_wrapper const& rhs)
{
return lhs.base_cursor - rhs.base_cursor;
}

}   
}   
}   
}   

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif
