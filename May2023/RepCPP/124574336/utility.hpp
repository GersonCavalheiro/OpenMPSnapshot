#ifndef  BOOST_SERIALIZATION_UTILITY_HPP
#define BOOST_SERIALIZATION_UTILITY_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <utility>
#include <boost/config.hpp>

#include <boost/type_traits/remove_const.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/mpl/and.hpp>

namespace boost {
namespace serialization {

template<class Archive, class F, class S>
inline void serialize(
Archive & ar,
std::pair<F, S> & p,
const unsigned int 
){
typedef typename boost::remove_const<F>::type typef;
ar & boost::serialization::make_nvp("first", const_cast<typef &>(p.first));
ar & boost::serialization::make_nvp("second", p.second);
}

template <class T, class U>
struct is_bitwise_serializable<std::pair<T,U> >
: public mpl::and_<is_bitwise_serializable< T >,is_bitwise_serializable<U> >
{
};

} 
} 

#endif 
