

#ifndef BOOST_MULTI_INDEX_DETAIL_NODE_TYPE_HPP
#define BOOST_MULTI_INDEX_DETAIL_NODE_TYPE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/mpl/bind.hpp>
#include <boost/mpl/reverse_iter_fold.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/multi_index_container_fwd.hpp>
#include <boost/multi_index/detail/header_holder.hpp>
#include <boost/multi_index/detail/index_node_base.hpp>
#include <boost/multi_index/detail/is_index_list.hpp>
#include <boost/static_assert.hpp>

namespace boost{

namespace multi_index{

namespace detail{



struct index_node_applier
{
template<typename IndexSpecifierIterator,typename Super>
struct apply
{
typedef typename mpl::deref<IndexSpecifierIterator>::type index_specifier;
typedef typename index_specifier::
BOOST_NESTED_TEMPLATE node_class<Super>::type type;
}; 
};

template<typename Value,typename IndexSpecifierList,typename Allocator>
struct multi_index_node_type
{
BOOST_STATIC_ASSERT(detail::is_index_list<IndexSpecifierList>::value);

typedef typename mpl::reverse_iter_fold<
IndexSpecifierList,
index_node_base<Value,Allocator>,
mpl::bind2<index_node_applier,mpl::_2,mpl::_1>
>::type type;
};

} 

} 

} 

#endif
