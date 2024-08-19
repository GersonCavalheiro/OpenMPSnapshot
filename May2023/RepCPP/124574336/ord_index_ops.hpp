

#ifndef BOOST_MULTI_INDEX_DETAIL_ORD_INDEX_OPS_HPP
#define BOOST_MULTI_INDEX_DETAIL_ORD_INDEX_OPS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/and.hpp>
#include <boost/multi_index/detail/promotes_arg.hpp>
#include <utility>

namespace boost{

namespace multi_index{

namespace detail{



template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline Node* ordered_index_find(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp)
{
typedef typename KeyFromValue::result_type key_type;

return ordered_index_find(
top,y,key,x,comp,
mpl::and_<
promotes_1st_arg<CompatibleCompare,CompatibleKey,key_type>,
promotes_2nd_arg<CompatibleCompare,key_type,CompatibleKey> >());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleCompare
>
inline Node* ordered_index_find(
Node* top,Node* y,const KeyFromValue& key,
const BOOST_DEDUCED_TYPENAME KeyFromValue::result_type& x,
const CompatibleCompare& comp,mpl::true_)
{
return ordered_index_find(top,y,key,x,comp,mpl::false_());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline Node* ordered_index_find(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp,mpl::false_)
{
Node* y0=y;

while (top){
if(!comp(key(top->value()),x)){
y=top;
top=Node::from_impl(top->left());
}
else top=Node::from_impl(top->right());
}

return (y==y0||comp(x,key(y->value())))?y0:y;
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline Node* ordered_index_lower_bound(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp)
{
typedef typename KeyFromValue::result_type key_type;

return ordered_index_lower_bound(
top,y,key,x,comp,
promotes_2nd_arg<CompatibleCompare,key_type,CompatibleKey>());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleCompare
>
inline Node* ordered_index_lower_bound(
Node* top,Node* y,const KeyFromValue& key,
const BOOST_DEDUCED_TYPENAME KeyFromValue::result_type& x,
const CompatibleCompare& comp,mpl::true_)
{
return ordered_index_lower_bound(top,y,key,x,comp,mpl::false_());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline Node* ordered_index_lower_bound(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp,mpl::false_)
{
while(top){
if(!comp(key(top->value()),x)){
y=top;
top=Node::from_impl(top->left());
}
else top=Node::from_impl(top->right());
}

return y;
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline Node* ordered_index_upper_bound(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp)
{
typedef typename KeyFromValue::result_type key_type;

return ordered_index_upper_bound(
top,y,key,x,comp,
promotes_1st_arg<CompatibleCompare,CompatibleKey,key_type>());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleCompare
>
inline Node* ordered_index_upper_bound(
Node* top,Node* y,const KeyFromValue& key,
const BOOST_DEDUCED_TYPENAME KeyFromValue::result_type& x,
const CompatibleCompare& comp,mpl::true_)
{
return ordered_index_upper_bound(top,y,key,x,comp,mpl::false_());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline Node* ordered_index_upper_bound(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp,mpl::false_)
{
while(top){
if(comp(x,key(top->value()))){
y=top;
top=Node::from_impl(top->left());
}
else top=Node::from_impl(top->right());
}

return y;
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline std::pair<Node*,Node*> ordered_index_equal_range(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp)
{
typedef typename KeyFromValue::result_type key_type;

return ordered_index_equal_range(
top,y,key,x,comp,
mpl::and_<
promotes_1st_arg<CompatibleCompare,CompatibleKey,key_type>,
promotes_2nd_arg<CompatibleCompare,key_type,CompatibleKey> >());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleCompare
>
inline std::pair<Node*,Node*> ordered_index_equal_range(
Node* top,Node* y,const KeyFromValue& key,
const BOOST_DEDUCED_TYPENAME KeyFromValue::result_type& x,
const CompatibleCompare& comp,mpl::true_)
{
return ordered_index_equal_range(top,y,key,x,comp,mpl::false_());
}

template<
typename Node,typename KeyFromValue,
typename CompatibleKey,typename CompatibleCompare
>
inline std::pair<Node*,Node*> ordered_index_equal_range(
Node* top,Node* y,const KeyFromValue& key,const CompatibleKey& x,
const CompatibleCompare& comp,mpl::false_)
{
while(top){
if(comp(key(top->value()),x)){
top=Node::from_impl(top->right());
}
else if(comp(x,key(top->value()))){
y=top;
top=Node::from_impl(top->left());
}
else{
return std::pair<Node*,Node*>(
ordered_index_lower_bound(
Node::from_impl(top->left()),top,key,x,comp,mpl::false_()),
ordered_index_upper_bound(
Node::from_impl(top->right()),y,key,x,comp,mpl::false_()));
}
}

return std::pair<Node*,Node*>(y,y);
}

} 

} 

} 

#endif
