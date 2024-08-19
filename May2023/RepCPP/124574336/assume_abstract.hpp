#ifndef BOOST_SERIALIZATION_ASSUME_ABSTRACT_HPP
#define BOOST_SERIALIZATION_ASSUME_ABSTRACT_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/type_traits/is_abstract.hpp>
#include <boost/mpl/bool_fwd.hpp>

#ifndef BOOST_NO_IS_ABSTRACT

#define BOOST_SERIALIZATION_ASSUME_ABSTRACT(T)

namespace boost {
namespace serialization {
template<class T>
struct is_abstract : boost::is_abstract< T > {} ;
} 
} 

#else

namespace boost {
namespace serialization {
template<class T>
struct is_abstract : boost::false_type {};
} 
} 

#define BOOST_SERIALIZATION_ASSUME_ABSTRACT(T)        \
namespace boost {                                     \
namespace serialization {                             \
template<>                                            \
struct is_abstract< T > : boost::true_type {};        \
template<>                                            \
struct is_abstract< const T > : boost::true_type {};  \
}}                                                    \


#endif 

#endif 
