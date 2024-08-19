



#ifndef BOOST_IOSTREAMS_ACCESS_CONTROL_HPP_INCLUDED
#define BOOST_IOSTREAMS_ACCESS_CONTROL_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/iostreams/detail/select.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost { namespace iostreams {

struct protected_ { };  
struct public_ { };     


namespace detail {

template<typename U>
struct prot_ : protected U 
{ 
prot_() { }
template<typename V> prot_(V v) : U(v) { }
};

template<typename U> struct pub_ : public U { 
pub_() { }
template<typename V> pub_(V v) : U(v) { }
};

template<typename T, typename Access>
struct access_control_base {
typedef int                                 bad_access_specifier;
typedef typename 
iostreams::select<  
::boost::is_same<
Access, protected_
>,                              prot_<T>,
::boost::is_same<
Access, public_
>,                              pub_<T>,
else_,                          bad_access_specifier
>::type                             type;
};

} 

template< typename T, typename Access,
typename Base = 
typename detail::access_control_base<T, Access>::type >
struct access_control : public Base { 
access_control() { }
template<typename U> explicit access_control(U u) : Base(u) { }
};


} } 

#endif 
