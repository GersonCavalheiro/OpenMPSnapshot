

#ifndef BOOST_MULTI_INDEX_SEQUENCED_INDEX_FWD_HPP
#define BOOST_MULTI_INDEX_SEQUENCED_INDEX_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/multi_index/tag.hpp>

namespace boost{

namespace multi_index{

namespace detail{

template<typename SuperMeta,typename TagList>
class sequenced_index;

template<
typename SuperMeta1,typename TagList1,
typename SuperMeta2,typename TagList2
>
bool operator==(
const sequenced_index<SuperMeta1,TagList1>& x,
const sequenced_index<SuperMeta2,TagList2>& y);

template<
typename SuperMeta1,typename TagList1,
typename SuperMeta2,typename TagList2
>
bool operator<(
const sequenced_index<SuperMeta1,TagList1>& x,
const sequenced_index<SuperMeta2,TagList2>& y);

template<
typename SuperMeta1,typename TagList1,
typename SuperMeta2,typename TagList2
>
bool operator!=(
const sequenced_index<SuperMeta1,TagList1>& x,
const sequenced_index<SuperMeta2,TagList2>& y);

template<
typename SuperMeta1,typename TagList1,
typename SuperMeta2,typename TagList2
>
bool operator>(
const sequenced_index<SuperMeta1,TagList1>& x,
const sequenced_index<SuperMeta2,TagList2>& y);

template<
typename SuperMeta1,typename TagList1,
typename SuperMeta2,typename TagList2
>
bool operator>=(
const sequenced_index<SuperMeta1,TagList1>& x,
const sequenced_index<SuperMeta2,TagList2>& y);

template<
typename SuperMeta1,typename TagList1,
typename SuperMeta2,typename TagList2
>
bool operator<=(
const sequenced_index<SuperMeta1,TagList1>& x,
const sequenced_index<SuperMeta2,TagList2>& y);

template<typename SuperMeta,typename TagList>
void swap(
sequenced_index<SuperMeta,TagList>& x,
sequenced_index<SuperMeta,TagList>& y);

} 



template <typename TagList=tag<> >
struct sequenced;

} 

} 

#endif
