#ifndef BOOST_SERIALIZATION_HASH_COLLECTIONS_LOAD_IMP_HPP
#define BOOST_SERIALIZATION_HASH_COLLECTIONS_LOAD_IMP_HPP

#if defined(_MSC_VER)
# pragma once
# pragma warning (disable : 4786) 
#endif




#include <boost/config.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>

namespace boost{
namespace serialization {
namespace stl {

template<class Archive, class Container, class InputFunction>
inline void load_hash_collection(Archive & ar, Container &s)
{
collection_size_type count;
collection_size_type bucket_count;
boost::serialization::item_version_type item_version(0);
boost::serialization::library_version_type library_version(
ar.get_library_version()
);
if(boost::serialization::library_version_type(6) != library_version){
ar >> BOOST_SERIALIZATION_NVP(count);
ar >> BOOST_SERIALIZATION_NVP(bucket_count);
}
else{
unsigned int c;
unsigned int bc;
ar >> BOOST_SERIALIZATION_NVP(c);
count = c;
ar >> BOOST_SERIALIZATION_NVP(bc);
bucket_count = bc;
}
if(boost::serialization::library_version_type(3) < library_version){
ar >> BOOST_SERIALIZATION_NVP(item_version);
}
s.clear();
#if ! defined(__MWERKS__)
s.resize(bucket_count);
#endif
InputFunction ifunc;
while(count-- > 0){
ifunc(ar, s, item_version);
}
}

} 
} 
} 

#endif 
