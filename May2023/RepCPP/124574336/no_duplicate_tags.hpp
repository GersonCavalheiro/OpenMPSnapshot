

#ifndef BOOST_MULTI_INDEX_DETAIL_NO_DUPLICATE_TAGS_HPP
#define BOOST_MULTI_INDEX_DETAIL_NO_DUPLICATE_TAGS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/fold.hpp>
#include <boost/mpl/set/set0.hpp>

namespace boost{

namespace multi_index{

namespace detail{



struct duplicate_tag_mark{};

struct duplicate_tag_marker
{
template <typename MplSet,typename Tag>
struct apply
{
typedef mpl::s_item<
typename mpl::if_<mpl::has_key<MplSet,Tag>,duplicate_tag_mark,Tag>::type,
MplSet
> type;
};
};

template<typename TagList>
struct no_duplicate_tags
{
typedef typename mpl::fold<
TagList,
mpl::set0<>,
duplicate_tag_marker
>::type aux;

BOOST_STATIC_CONSTANT(
bool,value=!(mpl::has_key<aux,duplicate_tag_mark>::value));
};



struct duplicate_tag_list_marker
{
template <typename MplSet,typename Index>
struct apply:mpl::fold<
BOOST_DEDUCED_TYPENAME Index::tag_list,
MplSet,
duplicate_tag_marker>
{
};
};

template<typename IndexList>
struct no_duplicate_tags_in_index_list
{
typedef typename mpl::fold<
IndexList,
mpl::set0<>,
duplicate_tag_list_marker
>::type aux;

BOOST_STATIC_CONSTANT(
bool,value=!(mpl::has_key<aux,duplicate_tag_mark>::value));
};

} 

} 

} 

#endif
