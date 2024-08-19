#ifndef BOOST_ARCHIVE_ITERATORS_BASE64_FROM_BINARY_HPP
#define BOOST_ARCHIVE_ITERATORS_BASE64_FROM_BINARY_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/assert.hpp>

#include <cstddef> 
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/iterator/transform_iterator.hpp>
#include <boost/archive/iterators/dataflow_exception.hpp>

namespace boost {
namespace archive {
namespace iterators {


namespace detail {

template<class CharType>
struct from_6_bit {
typedef CharType result_type;
CharType operator()(CharType t) const{
static const char * lookup_table =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789"
"+/";
BOOST_ASSERT(t < 64);
return lookup_table[static_cast<size_t>(t)];
}
};

} 


template<
class Base,
class CharType = typename boost::iterator_value<Base>::type
>
class base64_from_binary :
public transform_iterator<
detail::from_6_bit<CharType>,
Base
>
{
friend class boost::iterator_core_access;
typedef transform_iterator<
typename detail::from_6_bit<CharType>,
Base
> super_t;

public:
template<class T>
base64_from_binary(T start) :
super_t(
Base(static_cast< T >(start)),
detail::from_6_bit<CharType>()
)
{}
base64_from_binary(const base64_from_binary & rhs) :
super_t(
Base(rhs.base_reference()),
detail::from_6_bit<CharType>()
)
{}
};

} 
} 
} 

#endif 
