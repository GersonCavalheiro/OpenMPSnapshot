

#ifndef BOOST_IOSTREAMS_DETAIL_IOS_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_IOS_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/config/wide_streams.hpp>
#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# if !BOOST_WORKAROUND(__MWERKS__, <= 0x3003)
#  include <ios>
# else
#  include <istream>
#  include <ostream>
# endif
#else 
# include <exception>
# include <iosfwd>
#endif 

namespace boost { namespace iostreams { namespace detail {

#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES 
# define BOOST_IOSTREAMS_BASIC_IOS(ch, tr)  std::basic_ios< ch, tr >
# if !BOOST_WORKAROUND(__MWERKS__, <= 0x3003) && \
!BOOST_WORKAROUND(BOOST_BORLANDC, < 0x600) \


#define BOOST_IOS                std::ios
#define BOOST_IOSTREAMS_FAILURE  std::ios::failure

# else

#define BOOST_IOS                std::ios_base
#define BOOST_IOSTREAMS_FAILURE  std::ios_base::failure

# endif
#else 

#define BOOST_IOS                          std::ios
#define BOOST_IOSTREAMS_BASIC_IOS(ch, tr)  std::ios
#define BOOST_IOSTREAMS_FAILURE            boost::iostreams::detail::failure

class failure : std::exception {    
public:
explicit failure(const std::string& what_arg) : what_(what_arg) { }
const char* what() const { return what_.c_str(); }
private:
std::string what_;
};

#endif 

} } } 

#endif 
