

#ifndef BOOST_POLY_COLLECTION_DETAIL_SEGMENT_BACKEND_HPP
#define BOOST_POLY_COLLECTION_DETAIL_SEGMENT_BACKEND_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <cstddef>
#include <memory>
#include <utility>

namespace boost{

namespace poly_collection{

namespace detail{



template<typename Model,typename Allocator>
struct segment_backend
{
using segment_backend_unique_ptr=
std::unique_ptr<segment_backend,void(*)(segment_backend*)>;
using value_pointer=void*;
using const_value_pointer=const void*;
using base_iterator=typename Model::base_iterator;
using const_base_iterator=typename Model::const_base_iterator;
template<typename T>
using const_iterator=typename Model::template const_iterator<T>;
using base_sentinel=typename Model::base_sentinel;
using range=std::pair<base_iterator,base_sentinel>;

segment_backend()=default;
segment_backend(const segment_backend&)=delete;
segment_backend& operator=(const segment_backend&)=delete;

virtual                            ~segment_backend()=default;
virtual segment_backend_unique_ptr copy()const=0;
virtual segment_backend_unique_ptr copy(const Allocator&)const=0;
virtual segment_backend_unique_ptr empty_copy(const Allocator&)const=0;
virtual segment_backend_unique_ptr move(const Allocator&)=0;
virtual bool                       equal(const segment_backend&)const=0;

virtual Allocator     get_allocator()const noexcept=0;
virtual base_iterator begin()const noexcept=0;
virtual base_iterator end()const noexcept=0;
virtual bool          empty()const noexcept=0;
virtual std::size_t   size()const noexcept=0;
virtual std::size_t   max_size()const noexcept=0;
virtual std::size_t   capacity()const noexcept=0;
virtual base_sentinel reserve(std::size_t)=0;
virtual base_sentinel shrink_to_fit()=0;
virtual range         push_back(const_value_pointer)=0;
virtual range         push_back_move(value_pointer)=0;
virtual range         insert(const_base_iterator,const_value_pointer)=0;
virtual range         insert_move(const_base_iterator,value_pointer)=0;
virtual range         erase(const_base_iterator)=0;
virtual range         erase(const_base_iterator,const_base_iterator)=0;
virtual range         erase_till_end(const_base_iterator)=0;
virtual range         erase_from_begin(const_base_iterator)=0;
virtual base_sentinel clear()noexcept=0;
};

} 

} 

} 

#endif
