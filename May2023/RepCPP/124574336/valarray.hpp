#ifndef BOOST_SERIALIZATION_VALARAY_HPP
#define BOOST_SERIALIZATION_VALARAY_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <valarray>
#include <boost/config.hpp>
#include <boost/core/addressof.hpp>

#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array_wrapper.hpp>

#if defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)
#define STD _STLP_STD
#else
#define STD std
#endif

namespace boost {
namespace serialization {


template<class Archive, class U>
void save( Archive & ar, const STD::valarray<U> &t, const unsigned int  )
{
const collection_size_type count(t.size());
ar << BOOST_SERIALIZATION_NVP(count);
if (t.size()){
ar << serialization::make_array<const U, collection_size_type>(
static_cast<const U *>( boost::addressof(t[0]) ),
count
);
}
}

template<class Archive, class U>
void load( Archive & ar, STD::valarray<U> &t,  const unsigned int  )
{
collection_size_type count;
ar >> BOOST_SERIALIZATION_NVP(count);
t.resize(count);
if (t.size()){
ar >> serialization::make_array<U, collection_size_type>(
static_cast<U *>( boost::addressof(t[0]) ),
count
);
}
}

template<class Archive, class U>
inline void serialize( Archive & ar, STD::valarray<U> & t, const unsigned int file_version)
{
boost::serialization::split_free(ar, t, file_version);
}

} } 

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(STD::valarray)
#undef STD

#endif 
