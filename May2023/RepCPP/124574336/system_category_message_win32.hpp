#ifndef BOOST_SYSTEM_DETAIL_SYSTEM_CATEGORY_MESSAGE_WIN32_HPP_INCLUDED
#define BOOST_SYSTEM_DETAIL_SYSTEM_CATEGORY_MESSAGE_WIN32_HPP_INCLUDED


#include <boost/winapi/error_handling.hpp>
#include <boost/winapi/character_code_conversion.hpp>
#include <boost/winapi/local_memory.hpp>
#include <boost/config.hpp>
#include <cstdio>


namespace boost
{

namespace system
{

namespace detail
{

#if ( defined(_MSC_VER) && _MSC_VER < 1900 ) || ( defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR) )

inline char const * unknown_message_win32( int ev, char * buffer, std::size_t len )
{
# if defined( BOOST_MSVC )
#  pragma warning( push )
#  pragma warning( disable: 4996 )
# endif

_snprintf( buffer, len - 1, "Unknown error (%d)", ev );

buffer[ len - 1 ] = 0;
return buffer;

# if defined( BOOST_MSVC )
#  pragma warning( pop )
# endif
}

#else

inline char const * unknown_message_win32( int ev, char * buffer, std::size_t len )
{
std::snprintf( buffer, len, "Unknown error (%d)", ev );
return buffer;
}

#endif

inline boost::winapi::UINT_ message_cp_win32()
{
#if defined(BOOST_SYSTEM_USE_UTF8)

return boost::winapi::CP_UTF8_;

#else

return boost::winapi::CP_ACP_;

#endif
}

inline char const * system_category_message_win32( int ev, char * buffer, std::size_t len ) BOOST_NOEXCEPT
{
if( len == 0 )
{
return buffer;
}

if( len == 1 )
{
buffer[0] = 0;
return buffer;
}

boost::winapi::UINT_ const code_page = message_cp_win32();

int r = 0;

#if !defined(BOOST_NO_ANSI_APIS)

if( code_page == boost::winapi::CP_ACP_ )
{
using namespace boost::winapi;

DWORD_ retval = boost::winapi::FormatMessageA(
FORMAT_MESSAGE_FROM_SYSTEM_ | FORMAT_MESSAGE_IGNORE_INSERTS_,
NULL,
ev,
MAKELANGID_( LANG_NEUTRAL_, SUBLANG_DEFAULT_ ), 
buffer,
static_cast<DWORD_>( len ),
NULL
);

r = static_cast<int>( retval );
}
else

#endif

{
using namespace boost::winapi;

wchar_t * lpMsgBuf = 0;

DWORD_ retval = boost::winapi::FormatMessageW(
FORMAT_MESSAGE_ALLOCATE_BUFFER_ | FORMAT_MESSAGE_FROM_SYSTEM_ | FORMAT_MESSAGE_IGNORE_INSERTS_,
NULL,
ev,
MAKELANGID_( LANG_NEUTRAL_, SUBLANG_DEFAULT_ ), 
(LPWSTR_) &lpMsgBuf,
0,
NULL
);

if( retval != 0 )
{
r = boost::winapi::WideCharToMultiByte( code_page, 0, lpMsgBuf, -1, buffer, static_cast<int>( len ), NULL, NULL );
boost::winapi::LocalFree( lpMsgBuf );
if ( r != 0 ) --r; 
}
}

if( r == 0 )
{
return unknown_message_win32( ev, buffer, len );
}

while( r > 0 && ( buffer[ r-1 ] == '\n' || buffer[ r-1 ] == '\r' ) )
{
buffer[ --r ] = 0;
}

if( r > 0 && buffer[ r-1 ] == '.' )
{
buffer[ --r ] = 0;
}

return buffer;
}

struct local_free
{
void * p_;

~local_free()
{
boost::winapi::LocalFree( p_ );
}
};

inline std::string unknown_message_win32( int ev )
{
char buffer[ 38 ];
return unknown_message_win32( ev, buffer, sizeof( buffer ) );
}

inline std::string system_category_message_win32( int ev )
{
using namespace boost::winapi;

wchar_t * lpMsgBuf = 0;

DWORD_ retval = boost::winapi::FormatMessageW(
FORMAT_MESSAGE_ALLOCATE_BUFFER_ | FORMAT_MESSAGE_FROM_SYSTEM_ | FORMAT_MESSAGE_IGNORE_INSERTS_,
NULL,
ev,
MAKELANGID_( LANG_NEUTRAL_, SUBLANG_DEFAULT_ ), 
(LPWSTR_) &lpMsgBuf,
0,
NULL
);

if( retval == 0 )
{
return unknown_message_win32( ev );
}

local_free lf_ = { lpMsgBuf };
(void)lf_;

UINT_ const code_page = message_cp_win32();

int r = boost::winapi::WideCharToMultiByte( code_page, 0, lpMsgBuf, -1, 0, 0, NULL, NULL );

if( r == 0 )
{
return unknown_message_win32( ev );
}

std::string buffer( r, char() );

r = boost::winapi::WideCharToMultiByte( code_page, 0, lpMsgBuf, -1, &buffer[0], r, NULL, NULL );

if( r == 0 )
{
return unknown_message_win32( ev );
}

--r; 

while( r > 0 && ( buffer[ r-1 ] == '\n' || buffer[ r-1 ] == '\r' ) )
{
--r;
}

if( r > 0 && buffer[ r-1 ] == '.' )
{
--r;
}

buffer.resize( r );

return buffer;
}

} 

} 

} 

#endif 
