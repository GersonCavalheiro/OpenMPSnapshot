

#ifndef BOOST_MULTI_INDEX_DETAIL_HAS_TAG_HPP
#define BOOST_MULTI_INDEX_DETAIL_HAS_TAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/contains.hpp>

namespace boost{

namespace multi_index{

namespace detail{



template<typename Tag>
struct has_tag
{
template<typename Index>
struct apply:mpl::contains<BOOST_DEDUCED_TYPENAME Index::tag_list,Tag>
{
}; 
};

} 

} 

} 

#endif
