#ifndef BOOST_SERIALIZATION_UNORDERED_COLLECTIONS_LOAD_IMP_HPP
#define BOOST_SERIALIZATION_UNORDERED_COLLECTIONS_LOAD_IMP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
# pragma warning (disable : 4786) 
#endif





#include <boost/assert.hpp>
#include <cstddef> 
#include <boost/config.hpp> 
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif
#include <boost/detail/workaround.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>

namespace boost{
namespace serialization {
namespace stl {

template<class Archive, class Container, class InputFunction>
inline void load_unordered_collection(Archive & ar, Container &s)
{
collection_size_type count;
collection_size_type bucket_count;
boost::serialization::item_version_type item_version(0);
boost::serialization::library_version_type library_version(
ar.get_library_version()
);
ar >> BOOST_SERIALIZATION_NVP(count);
ar >> BOOST_SERIALIZATION_NVP(bucket_count);
if(boost::serialization::library_version_type(3) < library_version){
ar >> BOOST_SERIALIZATION_NVP(item_version);
}
s.clear();
s.rehash(bucket_count);
InputFunction ifunc;
while(count-- > 0){
ifunc(ar, s, item_version);
}
}

} 
} 
} 

#endif 
