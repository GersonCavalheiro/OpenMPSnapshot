

#ifndef BOOST_POLY_COLLECTION_FUNCTION_COLLECTION_FWD_HPP
#define BOOST_POLY_COLLECTION_FUNCTION_COLLECTION_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <memory>

namespace boost{

namespace poly_collection{

namespace detail{
template<typename Signature> struct function_model;
}

template<typename Signature>
using function_collection_value_type=
typename detail::function_model<Signature>::value_type;

template<
typename Signature,
typename Allocator=std::allocator<function_collection_value_type<Signature>>
>
class function_collection;

template<typename Signature,typename Allocator>
bool operator==(
const function_collection<Signature,Allocator>& x,
const function_collection<Signature,Allocator>& y);

template<typename Signature,typename Allocator>
bool operator!=(
const function_collection<Signature,Allocator>& x,
const function_collection<Signature,Allocator>& y);

template<typename Signature,typename Allocator>
void swap(
function_collection<Signature,Allocator>& x,
function_collection<Signature,Allocator>& y);

} 

using poly_collection::function_collection;

} 

#endif
