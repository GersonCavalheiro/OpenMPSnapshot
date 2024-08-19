#ifndef BOOST_CORE_CHECKED_DELETE_HPP
#define BOOST_CORE_CHECKED_DELETE_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/config.hpp>


namespace boost
{


template<class T> inline void checked_delete(T * x) BOOST_NOEXCEPT
{
typedef char type_must_be_complete[ sizeof(T)? 1: -1 ];
(void) sizeof(type_must_be_complete);
delete x;
}

template<class T> inline void checked_array_delete(T * x) BOOST_NOEXCEPT
{
typedef char type_must_be_complete[ sizeof(T)? 1: -1 ];
(void) sizeof(type_must_be_complete);
delete [] x;
}

template<class T> struct checked_deleter
{
typedef void result_type;
typedef T * argument_type;

void operator()(T * x) const BOOST_NOEXCEPT
{
boost::checked_delete(x);
}
};

template<class T> struct checked_array_deleter
{
typedef void result_type;
typedef T * argument_type;

void operator()(T * x) const BOOST_NOEXCEPT
{
boost::checked_array_delete(x);
}
};

} 

#endif  
