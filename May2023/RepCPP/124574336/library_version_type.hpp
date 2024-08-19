#ifndef BOOST_SERIALIZATION_LIBRARY_VERSION_TYPE_HPP
#define BOOST_SERIALIZATION_LIBRARY_VERSION_TYPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <cstring> 
#include <boost/cstdint.hpp> 
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/integer_traits.hpp>

namespace boost {
namespace serialization {

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4244 4267 )
#endif


class library_version_type {
private:
typedef uint_least16_t base_type;
base_type t;
public:
library_version_type(): t(0) {}
explicit library_version_type(const unsigned int & t_) : t(t_){
BOOST_ASSERT(t_ <= boost::integer_traits<base_type>::const_max);
}
library_version_type(const library_version_type & t_) :
t(t_.t)
{}
library_version_type & operator=(const library_version_type & rhs){
t = rhs.t;
return *this;
}
operator base_type () const {
return t;
}
operator base_type & (){
return t;
}
bool operator==(const library_version_type & rhs) const {
return t == rhs.t;
}
bool operator<(const library_version_type & rhs) const {
return t < rhs.t;
}
};

} 
} 

#endif 
