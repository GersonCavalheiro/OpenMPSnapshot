

#ifndef BOOST_POLY_COLLECTION_BASE_COLLECTION_FWD_HPP
#define BOOST_POLY_COLLECTION_BASE_COLLECTION_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <memory>

namespace boost{

namespace poly_collection{

template<typename Base,typename Allocator=std::allocator<Base>>
class base_collection;

template<typename Base,typename Allocator>
bool operator==(
const base_collection<Base,Allocator>& x,
const base_collection<Base,Allocator>& y);

template<typename Base,typename Allocator>
bool operator!=(
const base_collection<Base,Allocator>& x,
const base_collection<Base,Allocator>& y);

template<typename Base,typename Allocator>
void swap(
base_collection<Base,Allocator>& x,base_collection<Base,Allocator>& y);

} 

using poly_collection::base_collection;

} 

#endif
