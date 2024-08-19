

#include <boost/config.hpp>

#if defined(BOOST_HAS_UNISTD_H)
#include <unistd.h>
#else
#include <io.h>
#endif 

#include <boost/assert.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/wave/token_ids.hpp>
#include <boost/wave/cpplexer/re2clex/aq.hpp>
#include <boost/wave/cpplexer/re2clex/scanner.hpp>
#include <boost/wave/cpplexer/cpplexer_exceptions.hpp>

#include "idl_re.hpp"

#if defined(_MSC_VER) && !defined(__COMO__)
#pragma warning (disable: 4101)     
#pragma warning (disable: 4102)     
#endif

#define YYCTYPE   uchar
#define YYCURSOR  cursor
#define YYLIMIT   s->lim
#define YYMARKER  s->ptr
#define YYFILL(n) {cursor = fill(s, cursor);}

#define BOOST_WAVE_RET(i)    \
{ \
s->line += count_backslash_newlines(s, cursor); \
s->cur = cursor; \
return (i); \
} \


namespace boost {
namespace wave {
namespace idllexer {
namespace re2clex {

bool is_backslash(
boost::wave::cpplexer::re2clex::uchar *p, 
boost::wave::cpplexer::re2clex::uchar *end, int &len)
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


}   
}   
}   
}   
