#ifndef BOOST_SMART_PTR_DETAIL_SP_TYPEINFO_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_TYPEINFO_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>

#if defined( BOOST_NO_TYPEID ) || defined( BOOST_NO_STD_TYPEINFO )

#include <boost/core/typeinfo.hpp>

namespace boost
{

namespace detail
{

typedef boost::core::typeinfo sp_typeinfo_;

} 

} 

#define BOOST_SP_TYPEID_(T) BOOST_CORE_TYPEID(T)

#else 

#include <typeinfo>

namespace boost
{

namespace detail
{

typedef std::type_info sp_typeinfo_;

} 

} 

#define BOOST_SP_TYPEID_(T) typeid(T)

#endif 

#endif  
