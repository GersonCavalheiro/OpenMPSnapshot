

#ifndef BOOST_POLY_COLLECTION_FUNCTION_COLLECTION_HPP
#define BOOST_POLY_COLLECTION_FUNCTION_COLLECTION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/poly_collection/function_collection_fwd.hpp>
#include <boost/poly_collection/detail/function_model.hpp>
#include <boost/poly_collection/detail/poly_collection.hpp>
#include <utility>

namespace boost{

namespace poly_collection{

template<typename Signature,typename Allocator>
class function_collection:
public common_impl::poly_collection<
detail::function_model<Signature>,Allocator>
{
using base_type=common_impl::poly_collection<
detail::function_model<Signature>,Allocator>;

base_type&       base()noexcept{return *this;}
const base_type& base()const noexcept{return *this;}

public:
using base_type::base_type;

function_collection()=default;
function_collection(const function_collection& x)=default;
function_collection(function_collection&& x)=default;
function_collection& operator=(const function_collection& x)=default;
function_collection& operator=(function_collection&& x)=default;

template<typename S,typename A> 
friend bool operator==(
const function_collection<S,A>&,const function_collection<S,A>&);
};

template<typename Signature,typename Allocator>
bool operator==(
const function_collection<Signature,Allocator>& x,
const function_collection<Signature,Allocator>& y)
{
return x.base()==y.base();
}

template<typename Signature,typename Allocator>
bool operator!=(
const function_collection<Signature,Allocator>& x,
const function_collection<Signature,Allocator>& y)
{
return !(x==y);
}

template<typename Signature,typename Allocator>
void swap(
function_collection<Signature,Allocator>& x,
function_collection<Signature,Allocator>& y)
{
x.swap(y);
}

} 

using poly_collection::function_collection;

} 

#endif
