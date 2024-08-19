#ifndef BOOST_LOCALE_TIME_ZONE_HPP_INCLUDED
#define BOOST_LOCALE_TIME_ZONE_HPP_INCLUDED

#include <boost/locale/config.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif

#include <string>


namespace boost {
namespace locale {

namespace time_zone {
BOOST_LOCALE_DECL std::string global();
BOOST_LOCALE_DECL std::string global(std::string const &new_tz);
}


} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


#endif

