

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_KEY_EXTRACTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_KEY_EXTRACTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {
namespace container_adaptor {
namespace detail {


template < class T >
struct key_from_pair_extractor 
{
typedef T argument_type;
typedef BOOST_DEDUCED_TYPENAME T::first_type result_type;

result_type operator()( const T & p ) { return p.first; }
};

} 
} 
} 
} 


#endif 


