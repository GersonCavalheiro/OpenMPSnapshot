#ifndef BOOST_SERIALIZATION_BINARY_OBJECT_HPP
#define BOOST_SERIALIZATION_BINARY_OBJECT_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>

#include <cstddef> 
#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/preprocessor/stringize.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/wrapper.hpp>

namespace boost {
namespace serialization {

struct binary_object :
public wrapper_traits<nvp<const binary_object> >
{
void const * m_t;
std::size_t m_size;
template<class Archive>
void save(Archive & ar, const unsigned int ) const {
ar.save_binary(m_t, m_size);
}
template<class Archive>
void load(Archive & ar, const unsigned int ) const {
ar.load_binary(const_cast<void *>(m_t), m_size);
}
BOOST_SERIALIZATION_SPLIT_MEMBER()
binary_object & operator=(const binary_object & rhs) {
m_t = rhs.m_t;
m_size = rhs.m_size;
return *this;
}
binary_object(const void * const t, std::size_t size) :
m_t(t),
m_size(size)
{}
binary_object(const binary_object & rhs) :
m_t(rhs.m_t),
m_size(rhs.m_size)
{}
};

inline
const binary_object
make_binary_object(const void * t, std::size_t size){
return binary_object(t, size);
}

} 
} 

#endif 
