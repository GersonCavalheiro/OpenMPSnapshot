



#ifndef BOOST_SYSTEM_CYGWIN_ERROR_HPP
#define BOOST_SYSTEM_CYGWIN_ERROR_HPP

#include <boost/config/pragma_message.hpp>

#if !defined(BOOST_ALLOW_DEPRECATED_HEADERS)
BOOST_PRAGMA_MESSAGE("This header is deprecated and is slated for removal."
" If you want it retained, please open an issue in github.com/boostorg/system.")
#endif


# ifdef __CYGWIN__

#include <boost/system/error_code.hpp>

namespace boost
{
namespace system
{


namespace cygwin_error
{
enum cygwin_errno
{
no_net = ENONET,
no_package = ENOPKG,
no_share = ENOSHARE
};
}  

template<> struct is_error_code_enum<cygwin_error::cygwin_errno>
{ static const bool value = true; };

namespace cygwin_error
{
inline error_code make_error_code( cygwin_errno e )
{ return error_code( e, system_category() ); }
}
}
}

#endif  

#endif  
