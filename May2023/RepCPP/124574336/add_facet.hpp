


#ifndef BOOST_IOSTREAMS_DETAIL_ADD_FACET_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_ADD_FACET_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>


#if (defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)) && \
defined(_STLP_NO_OWN_IOSTREAMS) \

#  if (defined(_YVALS) && !defined(__IBMCPP__)) || defined(_CPPLIB_VER)
#    define BOOST_IOSTREMS_STLPORT_WITH_OLD_DINKUMWARE
#  endif
#endif

namespace boost { namespace iostreams { namespace detail {

template<class Facet>
inline std::locale add_facet(const std::locale &l, Facet * f)
{
return
#if BOOST_WORKAROUND(BOOST_DINKUMWARE_STDLIB, == 1) || \
defined(BOOST_IOSTREMS_STLPORT_WITH_OLD_DINKUMWARE) \

std::locale(std::_Addfac(l, f));
#else
std::locale(l, f);
#endif
}

} } } 

#endif 
