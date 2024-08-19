

#ifndef BOOST_MULTI_INDEX_DETAIL_RND_INDEX_OPS_HPP
#define BOOST_MULTI_INDEX_DETAIL_RND_INDEX_OPS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>
#include <boost/multi_index/detail/rnd_index_ptr_array.hpp>

namespace boost{

namespace multi_index{

namespace detail{



template<typename Node,typename Allocator,typename Predicate>
Node* random_access_index_remove(
random_access_index_ptr_array<Allocator>& ptrs,Predicate pred)
{
typedef typename Node::value_type value_type;
typedef typename Node::impl_ptr_pointer impl_ptr_pointer;

impl_ptr_pointer first=ptrs.begin(),
res=first,
last=ptrs.end();
for(;first!=last;++first){
if(!pred(
const_cast<const value_type&>(Node::from_impl(*first)->value()))){
if(first!=res){
std::swap(*first,*res);
(*first)->up()=first;
(*res)->up()=res;
}
++res;
}
}
return Node::from_impl(*res);
}

template<typename Node,typename Allocator,class BinaryPredicate>
Node* random_access_index_unique(
random_access_index_ptr_array<Allocator>& ptrs,BinaryPredicate binary_pred)
{
typedef typename Node::value_type       value_type;
typedef typename Node::impl_ptr_pointer impl_ptr_pointer;

impl_ptr_pointer first=ptrs.begin(),
res=first,
last=ptrs.end();
if(first!=last){
for(;++first!=last;){
if(!binary_pred(
const_cast<const value_type&>(Node::from_impl(*res)->value()),
const_cast<const value_type&>(Node::from_impl(*first)->value()))){
++res;
if(first!=res){
std::swap(*first,*res);
(*first)->up()=first;
(*res)->up()=res;
}
}
}
++res;
}
return Node::from_impl(*res);
}

template<typename Node,typename Allocator,typename Compare>
void random_access_index_inplace_merge(
const Allocator& al,
random_access_index_ptr_array<Allocator>& ptrs,
BOOST_DEDUCED_TYPENAME Node::impl_ptr_pointer first1,Compare comp)
{
typedef typename Node::value_type       value_type;
typedef typename Node::impl_pointer     impl_pointer;
typedef typename Node::impl_ptr_pointer impl_ptr_pointer;

auto_space<impl_pointer,Allocator> spc(al,ptrs.size());

impl_ptr_pointer first0=ptrs.begin(),
last0=first1,
last1=ptrs.end(),
out=spc.data();
while(first0!=last0&&first1!=last1){
if(comp(
const_cast<const value_type&>(Node::from_impl(*first1)->value()),
const_cast<const value_type&>(Node::from_impl(*first0)->value()))){
*out++=*first1++;
}
else{
*out++=*first0++;
}
}
std::copy(&*first0,&*last0,&*out);
std::copy(&*first1,&*last1,&*out);

first1=ptrs.begin();
out=spc.data();
while(first1!=last1){
*first1=*out++;
(*first1)->up()=first1;
++first1;
}
}





template<typename Node,typename Compare>
struct random_access_index_sort_compare
{
typedef typename Node::impl_pointer first_argument_type;
typedef typename Node::impl_pointer second_argument_type;
typedef bool                        result_type;

random_access_index_sort_compare(Compare comp_=Compare()):comp(comp_){}

bool operator()(
typename Node::impl_pointer x,typename Node::impl_pointer y)const
{
typedef typename Node::value_type value_type;

return comp(
const_cast<const value_type&>(Node::from_impl(x)->value()),
const_cast<const value_type&>(Node::from_impl(y)->value()));
}

private:
Compare comp;
};

template<typename Node,typename Allocator,class Compare>
void random_access_index_sort(
const Allocator& al,
random_access_index_ptr_array<Allocator>& ptrs,
Compare comp)
{


if(ptrs.size()<=1)return;

typedef typename Node::impl_pointer       impl_pointer;
typedef typename Node::impl_ptr_pointer   impl_ptr_pointer;
typedef random_access_index_sort_compare<
Node,Compare>                           ptr_compare;

impl_ptr_pointer   first=ptrs.begin();
impl_ptr_pointer   last=ptrs.end();
auto_space<
impl_pointer,
Allocator>       spc(al,ptrs.size());
impl_ptr_pointer   buf=spc.data();

std::copy(&*first,&*last,&*buf);
std::stable_sort(&*buf,&*buf+ptrs.size(),ptr_compare(comp));

while(first!=last){
*first=*buf++;
(*first)->up()=first;
++first;
}
}

} 

} 

} 

#endif
