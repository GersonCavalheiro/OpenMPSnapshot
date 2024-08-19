

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_ITERATOR_FACADE_CONVERTERS_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_ITERATOR_FACADE_CONVERTERS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {
namespace container_adaptor {


namespace support {


template
<
class Iterator,
class ConstIterator
>
struct iterator_facade_to_base
{
BOOST_DEDUCED_TYPENAME Iterator::base_type operator()(Iterator iter) const
{
return iter.base();
}

BOOST_DEDUCED_TYPENAME ConstIterator::base_type operator()(ConstIterator iter) const
{
return iter.base();
}
};

#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

template
<
class Iterator
>
struct iterator_facade_to_base<Iterator,Iterator>
{
BOOST_DEDUCED_TYPENAME Iterator::base_type operator()(Iterator iter) const
{
return iter.base();
}
};

#endif 

#undef BOOST_BIMAP_CONTAINER_ADAPTOR_IMPLEMENT_CONVERT_FACADE_FUNCTION


} 
} 
} 
} 


#endif 
