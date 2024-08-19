

#ifndef BOOST_MULTI_INDEX_DETAIL_ORD_INDEX_IMPL_FWD_HPP
#define BOOST_MULTI_INDEX_DETAIL_ORD_INDEX_IMPL_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

namespace boost{

namespace multi_index{

namespace detail{

template<
typename KeyFromValue,typename Compare,
typename SuperMeta,typename TagList,typename Category,typename AugmentPolicy
>
class ordered_index;

template<
typename KeyFromValue1,typename Compare1,
typename SuperMeta1,typename TagList1,typename Category1,
typename AugmentPolicy1,
typename KeyFromValue2,typename Compare2,
typename SuperMeta2,typename TagList2,typename Category2,
typename AugmentPolicy2
>
bool operator==(
const ordered_index<
KeyFromValue1,Compare1,SuperMeta1,TagList1,Category1,AugmentPolicy1>& x,
const ordered_index<
KeyFromValue2,Compare2,SuperMeta2,TagList2,Category2,AugmentPolicy2>& y);

template<
typename KeyFromValue1,typename Compare1,
typename SuperMeta1,typename TagList1,typename Category1,
typename AugmentPolicy1,
typename KeyFromValue2,typename Compare2,
typename SuperMeta2,typename TagList2,typename Category2,
typename AugmentPolicy2
>
bool operator<(
const ordered_index<
KeyFromValue1,Compare1,SuperMeta1,TagList1,Category1,AugmentPolicy1>& x,
const ordered_index<
KeyFromValue2,Compare2,SuperMeta2,TagList2,Category2,AugmentPolicy2>& y);

template<
typename KeyFromValue1,typename Compare1,
typename SuperMeta1,typename TagList1,typename Category1,
typename AugmentPolicy1,
typename KeyFromValue2,typename Compare2,
typename SuperMeta2,typename TagList2,typename Category2,
typename AugmentPolicy2
>
bool operator!=(
const ordered_index<
KeyFromValue1,Compare1,SuperMeta1,TagList1,Category1,AugmentPolicy1>& x,
const ordered_index<
KeyFromValue2,Compare2,SuperMeta2,TagList2,Category2,AugmentPolicy2>& y);

template<
typename KeyFromValue1,typename Compare1,
typename SuperMeta1,typename TagList1,typename Category1,
typename AugmentPolicy1,
typename KeyFromValue2,typename Compare2,
typename SuperMeta2,typename TagList2,typename Category2,
typename AugmentPolicy2
>
bool operator>(
const ordered_index<
KeyFromValue1,Compare1,SuperMeta1,TagList1,Category1,AugmentPolicy1>& x,
const ordered_index<
KeyFromValue2,Compare2,SuperMeta2,TagList2,Category2,AugmentPolicy2>& y);

template<
typename KeyFromValue1,typename Compare1,
typename SuperMeta1,typename TagList1,typename Category1,
typename AugmentPolicy1,
typename KeyFromValue2,typename Compare2,
typename SuperMeta2,typename TagList2,typename Category2,
typename AugmentPolicy2
>
bool operator>=(
const ordered_index<
KeyFromValue1,Compare1,SuperMeta1,TagList1,Category1,AugmentPolicy1>& x,
const ordered_index<
KeyFromValue2,Compare2,SuperMeta2,TagList2,Category2,AugmentPolicy2>& y);

template<
typename KeyFromValue1,typename Compare1,
typename SuperMeta1,typename TagList1,typename Category1,
typename AugmentPolicy1,
typename KeyFromValue2,typename Compare2,
typename SuperMeta2,typename TagList2,typename Category2,
typename AugmentPolicy2
>
bool operator<=(
const ordered_index<
KeyFromValue1,Compare1,SuperMeta1,TagList1,Category1,AugmentPolicy1>& x,
const ordered_index<
KeyFromValue2,Compare2,SuperMeta2,TagList2,Category2,AugmentPolicy2>& y);

template<
typename KeyFromValue,typename Compare,
typename SuperMeta,typename TagList,typename Category,typename AugmentPolicy
>
void swap(
ordered_index<
KeyFromValue,Compare,SuperMeta,TagList,Category,AugmentPolicy>& x,
ordered_index<
KeyFromValue,Compare,SuperMeta,TagList,Category,AugmentPolicy>& y);

} 

} 

} 

#endif
