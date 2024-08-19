#ifndef BOOST_DETAIL_SP_TYPEINFO_HPP_INCLUDED
#define BOOST_DETAIL_SP_TYPEINFO_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/core/typeinfo.hpp>
#include <boost/config/header_deprecated.hpp>

BOOST_HEADER_DEPRECATED( "<boost/core/typeinfo.hpp>" )

namespace boost
{

namespace detail
{

typedef boost::core::typeinfo sp_typeinfo;

} 

} 

#define BOOST_SP_TYPEID(T) BOOST_CORE_TYPEID(T)

#endif  
