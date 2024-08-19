

#ifndef BOOST_MULTI_INDEX_FWD_HPP
#define BOOST_MULTI_INDEX_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/indexed_by.hpp>
#include <boost/multi_index/ordered_index_fwd.hpp>
#include <memory>

namespace boost{

namespace multi_index{



template<
typename Value,
typename IndexSpecifierList=indexed_by<ordered_unique<identity<Value> > >,
typename Allocator=std::allocator<Value> >
class multi_index_container;

template<typename MultiIndexContainer,int N>
struct nth_index;

template<typename MultiIndexContainer,typename Tag>
struct index;

template<typename MultiIndexContainer,int N>
struct nth_index_iterator;

template<typename MultiIndexContainer,int N>
struct nth_index_const_iterator;

template<typename MultiIndexContainer,typename Tag>
struct index_iterator;

template<typename MultiIndexContainer,typename Tag>
struct index_const_iterator;



template<
typename Value1,typename IndexSpecifierList1,typename Allocator1,
typename Value2,typename IndexSpecifierList2,typename Allocator2
>
bool operator==(
const multi_index_container<Value1,IndexSpecifierList1,Allocator1>& x,
const multi_index_container<Value2,IndexSpecifierList2,Allocator2>& y);

template<
typename Value1,typename IndexSpecifierList1,typename Allocator1,
typename Value2,typename IndexSpecifierList2,typename Allocator2
>
bool operator<(
const multi_index_container<Value1,IndexSpecifierList1,Allocator1>& x,
const multi_index_container<Value2,IndexSpecifierList2,Allocator2>& y);

template<
typename Value1,typename IndexSpecifierList1,typename Allocator1,
typename Value2,typename IndexSpecifierList2,typename Allocator2
>
bool operator!=(
const multi_index_container<Value1,IndexSpecifierList1,Allocator1>& x,
const multi_index_container<Value2,IndexSpecifierList2,Allocator2>& y);

template<
typename Value1,typename IndexSpecifierList1,typename Allocator1,
typename Value2,typename IndexSpecifierList2,typename Allocator2
>
bool operator>(
const multi_index_container<Value1,IndexSpecifierList1,Allocator1>& x,
const multi_index_container<Value2,IndexSpecifierList2,Allocator2>& y);

template<
typename Value1,typename IndexSpecifierList1,typename Allocator1,
typename Value2,typename IndexSpecifierList2,typename Allocator2
>
bool operator>=(
const multi_index_container<Value1,IndexSpecifierList1,Allocator1>& x,
const multi_index_container<Value2,IndexSpecifierList2,Allocator2>& y);

template<
typename Value1,typename IndexSpecifierList1,typename Allocator1,
typename Value2,typename IndexSpecifierList2,typename Allocator2
>
bool operator<=(
const multi_index_container<Value1,IndexSpecifierList1,Allocator1>& x,
const multi_index_container<Value2,IndexSpecifierList2,Allocator2>& y);

template<typename Value,typename IndexSpecifierList,typename Allocator>
void swap(
multi_index_container<Value,IndexSpecifierList,Allocator>& x,
multi_index_container<Value,IndexSpecifierList,Allocator>& y);

} 



using multi_index::multi_index_container;

} 

#endif
