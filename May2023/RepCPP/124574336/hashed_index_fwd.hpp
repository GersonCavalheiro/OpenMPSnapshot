

#ifndef BOOST_MULTI_INDEX_HASHED_INDEX_FWD_HPP
#define BOOST_MULTI_INDEX_HASHED_INDEX_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/multi_index/detail/hash_index_args.hpp>

namespace boost{

namespace multi_index{

namespace detail{

template<
typename KeyFromValue,typename Hash,typename Pred,
typename SuperMeta,typename TagList,typename Category
>
class hashed_index;

template<
typename KeyFromValue,typename Hash,typename Pred,
typename SuperMeta,typename TagList,typename Category
>
bool operator==(
const hashed_index<KeyFromValue,Hash,Pred,SuperMeta,TagList,Category>& x,
const hashed_index<KeyFromValue,Hash,Pred,SuperMeta,TagList,Category>& y);

template<
typename KeyFromValue,typename Hash,typename Pred,
typename SuperMeta,typename TagList,typename Category
>
bool operator!=(
const hashed_index<KeyFromValue,Hash,Pred,SuperMeta,TagList,Category>& x,
const hashed_index<KeyFromValue,Hash,Pred,SuperMeta,TagList,Category>& y);

template<
typename KeyFromValue,typename Hash,typename Pred,
typename SuperMeta,typename TagList,typename Category
>
void swap(
hashed_index<KeyFromValue,Hash,Pred,SuperMeta,TagList,Category>& x,
hashed_index<KeyFromValue,Hash,Pred,SuperMeta,TagList,Category>& y);

} 



template<
typename Arg1,typename Arg2=mpl::na,
typename Arg3=mpl::na,typename Arg4=mpl::na
>
struct hashed_unique;

template<
typename Arg1,typename Arg2=mpl::na,
typename Arg3=mpl::na,typename Arg4=mpl::na
>
struct hashed_non_unique;

} 

} 

#endif
