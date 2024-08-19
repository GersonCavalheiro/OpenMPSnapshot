

#ifndef BOOST_MULTI_INDEX_DETAIL_BASE_TYPE_HPP
#define BOOST_MULTI_INDEX_DETAIL_BASE_TYPE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/size.hpp>
#include <boost/multi_index/detail/index_base.hpp>
#include <boost/multi_index/detail/is_index_list.hpp>
#include <boost/static_assert.hpp>

namespace boost{

namespace multi_index{

namespace detail{



struct index_applier
{
template<typename IndexSpecifierMeta,typename SuperMeta>
struct apply
{
typedef typename IndexSpecifierMeta::type            index_specifier;
typedef typename index_specifier::
BOOST_NESTED_TEMPLATE index_class<SuperMeta>::type type;
}; 
};

template<int N,typename Value,typename IndexSpecifierList,typename Allocator>
struct nth_layer
{
BOOST_STATIC_CONSTANT(int,length=mpl::size<IndexSpecifierList>::value);

typedef typename  mpl::eval_if_c<
N==length,
mpl::identity<index_base<Value,IndexSpecifierList,Allocator> >,
mpl::apply2<
index_applier,
mpl::at_c<IndexSpecifierList,N>,
nth_layer<N+1,Value,IndexSpecifierList,Allocator>
>
>::type type;
};

template<typename Value,typename IndexSpecifierList,typename Allocator>
struct multi_index_base_type:nth_layer<0,Value,IndexSpecifierList,Allocator>
{
BOOST_STATIC_ASSERT(detail::is_index_list<IndexSpecifierList>::value);
};

} 

} 

} 

#endif
