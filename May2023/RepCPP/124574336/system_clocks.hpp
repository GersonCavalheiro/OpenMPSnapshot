






#ifndef BOOST_CHRONO_SYSTEM_CLOCKS_HPP
#define BOOST_CHRONO_SYSTEM_CLOCKS_HPP

#include <boost/chrono/config.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/chrono/time_point.hpp>
#include <boost/chrono/detail/system.hpp>
#include <boost/chrono/clock_string.hpp>

#include <ctime>

# if defined( BOOST_CHRONO_POSIX_API )
#   if ! defined(CLOCK_REALTIME) && ! defined (__hpux__)
#     error <time.h> does not supply CLOCK_REALTIME
#   endif
# endif

#ifdef BOOST_CHRONO_WINDOWS_API
# define BOOST_SYSTEM_CLOCK_DURATION boost::chrono::duration<boost::int_least64_t, ratio<BOOST_RATIO_INTMAX_C(1), BOOST_RATIO_INTMAX_C(10000000)> >
#else
# define BOOST_SYSTEM_CLOCK_DURATION boost::chrono::nanoseconds
#endif

#ifndef BOOST_CHRONO_HEADER_ONLY
#include <boost/config/abi_prefix.hpp> 
#endif



namespace boost {
namespace chrono {

class system_clock;
#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
class steady_clock;
#endif

#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
typedef steady_clock high_resolution_clock;  
#else
typedef system_clock high_resolution_clock;  
#endif




class BOOST_CHRONO_DECL system_clock
{
public:
typedef BOOST_SYSTEM_CLOCK_DURATION          duration;
typedef duration::rep                        rep;
typedef duration::period                     period;
typedef chrono::time_point<system_clock>     time_point;
BOOST_STATIC_CONSTEXPR bool is_steady =             false;

static BOOST_CHRONO_INLINE time_point  now() BOOST_NOEXCEPT;
#if !defined BOOST_CHRONO_DONT_PROVIDE_HYBRID_ERROR_HANDLING
static BOOST_CHRONO_INLINE time_point  now(system::error_code & ec);
#endif

static BOOST_CHRONO_INLINE std::time_t to_time_t(const time_point& t) BOOST_NOEXCEPT;
static BOOST_CHRONO_INLINE time_point  from_time_t(std::time_t t) BOOST_NOEXCEPT;
};



#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
class BOOST_CHRONO_DECL steady_clock
{
public:
typedef nanoseconds                          duration;
typedef duration::rep                        rep;
typedef duration::period                     period;
typedef chrono::time_point<steady_clock>  time_point;
BOOST_STATIC_CONSTEXPR bool is_steady =             true;

static BOOST_CHRONO_INLINE time_point  now() BOOST_NOEXCEPT;
#if !defined BOOST_CHRONO_DONT_PROVIDE_HYBRID_ERROR_HANDLING
static BOOST_CHRONO_INLINE time_point  now(system::error_code & ec);
#endif
};
#endif



template<class CharT>
struct clock_string<system_clock, CharT>
{
static std::basic_string<CharT> name()
{
static const CharT u[] =
{ 's', 'y', 's', 't', 'e', 'm', '_', 'c', 'l', 'o', 'c', 'k' };
static const std::basic_string<CharT> str(u, u + sizeof(u)
/ sizeof(u[0]));
return str;
}
static std::basic_string<CharT> since()
{
static const CharT
u[] =
{ ' ', 's', 'i', 'n', 'c', 'e', ' ', 'J', 'a', 'n', ' ', '1', ',', ' ', '1', '9', '7', '0' };
static const std::basic_string<CharT> str(u, u + sizeof(u)
/ sizeof(u[0]));
return str;
}
};

#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY

template<class CharT>
struct clock_string<steady_clock, CharT>
{
static std::basic_string<CharT> name()
{
static const CharT
u[] =
{ 's', 't', 'e', 'a', 'd', 'y', '_', 'c', 'l', 'o', 'c', 'k' };
static const std::basic_string<CharT> str(u, u + sizeof(u)
/ sizeof(u[0]));
return str;
}
static std::basic_string<CharT> since()
{
const CharT u[] =
{ ' ', 's', 'i', 'n', 'c', 'e', ' ', 'b', 'o', 'o', 't' };
const std::basic_string<CharT> str(u, u + sizeof(u) / sizeof(u[0]));
return str;
}
};

#endif

} 
} 

#ifndef BOOST_CHRONO_HEADER_ONLY
#include <boost/config/abi_suffix.hpp> 
#else
#include <boost/chrono/detail/inlined/chrono.hpp>
#endif

#endif 
