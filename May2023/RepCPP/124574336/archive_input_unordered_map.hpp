#ifndef  BOOST_SERIALIZATION_ARCHIVE_INPUT_UNORDERED_MAP_HPP
#define BOOST_SERIALIZATION_ARCHIVE_INPUT_UNORDERED_MAP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace serialization {
namespace stl {

template<class Archive, class Container>
struct archive_input_unordered_map
{
inline void operator()(
Archive &ar,
Container &s,
const unsigned int v
){
typedef typename Container::value_type type;
detail::stack_construct<Archive, type> t(ar, v);
ar >> boost::serialization::make_nvp("item", t.reference());
std::pair<typename Container::const_iterator, bool> result =
s.insert(boost::move(t.reference()));
if(result.second){
ar.reset_object_address(
& (result.first->second),
& t.reference().second
);
}
}
};

template<class Archive, class Container>
struct archive_input_unordered_multimap
{
inline void operator()(
Archive &ar,
Container &s,
const unsigned int v
){
typedef typename Container::value_type type;
detail::stack_construct<Archive, type> t(ar, v);
ar >> boost::serialization::make_nvp("item", t.reference());
typename Container::const_iterator result =
s.insert(t.reference());
ar.reset_object_address(
& result->second,
& t.reference()
);
}
};

} 
} 
} 

#endif 
