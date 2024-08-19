

#ifndef BOOST_POLY_COLLECTION_ANY_COLLECTION_FWD_HPP
#define BOOST_POLY_COLLECTION_ANY_COLLECTION_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <memory>

namespace boost{

namespace poly_collection{

namespace detail{
template<typename Concept> struct any_model;
}

template<typename Concept>
using any_collection_value_type=
typename detail::any_model<Concept>::value_type;

template<
typename Concept,
typename Allocator=std::allocator<any_collection_value_type<Concept>>
>
class any_collection;

template<typename Concept,typename Allocator>
bool operator==(
const any_collection<Concept,Allocator>& x,
const any_collection<Concept,Allocator>& y);

template<typename Concept,typename Allocator>
bool operator!=(
const any_collection<Concept,Allocator>& x,
const any_collection<Concept,Allocator>& y);

template<typename Concept,typename Allocator>
void swap(
any_collection<Concept,Allocator>& x,any_collection<Concept,Allocator>& y);

} 

using poly_collection::any_collection;

} 

#endif
