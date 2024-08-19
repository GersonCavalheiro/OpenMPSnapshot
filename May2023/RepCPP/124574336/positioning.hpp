


#ifndef BOOST_IOSTREAMS_POSITIONING_HPP_INCLUDED
#define BOOST_IOSTREAMS_POSITIONING_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/integer_traits.hpp>
#include <boost/iostreams/detail/config/codecvt.hpp> 
#include <boost/iostreams/detail/config/fpos.hpp>
#include <boost/iostreams/detail/ios.hpp> 

#include <boost/iostreams/detail/config/disable_warnings.hpp> 

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::fpos_t; }
#endif

namespace boost { namespace iostreams {


typedef boost::intmax_t stream_offset;


inline std::streamoff stream_offset_to_streamoff(stream_offset off)
{ return static_cast<stream_offset>(off); }


# ifndef BOOST_IOSTREAMS_HAS_DINKUMWARE_FPOS

inline std::streampos offset_to_position(stream_offset off) { return off; }

# else 

inline std::streampos offset_to_position(stream_offset off)
{ return std::streampos(std::mbstate_t(), off); }

# endif 


template<typename PosType> 
inline stream_offset position_to_offset(PosType pos)
{ return std::streamoff(pos); }

# ifndef BOOST_IOSTREAMS_HAS_DINKUMWARE_FPOS

inline stream_offset position_to_offset(std::streampos pos) { return pos; }

# else 


inline stream_offset fpos_t_to_offset(std::fpos_t pos)
{
#  if defined(_POSIX_) || (_INTEGRAL_MAX_BITS >= 64) || defined(__IBMCPP__)
return pos;
#  else
return BOOST_IOSTREAMS_FPOSOFF(pos);
#  endif
}

inline std::fpos_t streampos_to_fpos_t(std::streampos pos)
{
#  if defined (_CPPLIB_VER) || defined(__IBMCPP__)
return pos.seekpos();
#  else
return pos.get_fpos_t();
#  endif
}

inline stream_offset position_to_offset(std::streampos pos)
{
return fpos_t_to_offset(streampos_to_fpos_t(pos)) +
static_cast<stream_offset>(
static_cast<std::streamoff>(pos) -
BOOST_IOSTREAMS_FPOSOFF(streampos_to_fpos_t(pos))
);
}

# endif 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp> 

#endif 
