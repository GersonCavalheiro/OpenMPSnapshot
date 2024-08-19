



#ifndef BOOST_FILESYSTEM3_CONVENIENCE_HPP
#define BOOST_FILESYSTEM3_CONVENIENCE_HPP

#include <boost/config.hpp>

# if defined( BOOST_NO_STD_WSTRING )
#   error Configuration not supported: Boost.Filesystem V3 and later requires std::wstring support
# endif

#include <boost/filesystem/operations.hpp>
#include <boost/system/error_code.hpp>

#include <boost/config/abi_prefix.hpp> 

namespace boost
{
namespace filesystem
{

# ifndef BOOST_FILESYSTEM_NO_DEPRECATED

inline std::string extension(const path & p)
{
return p.extension().string();
}

inline std::string basename(const path & p)
{
return p.stem().string();
}

inline path change_extension( const path & p, const path & new_extension )
{
path new_p( p );
new_p.replace_extension( new_extension );
return new_p;
}

# endif


} 
} 

#include <boost/config/abi_suffix.hpp> 
#endif 
