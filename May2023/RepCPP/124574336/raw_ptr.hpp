

#ifndef BOOST_MULTI_INDEX_DETAIL_RAW_PTR_HPP
#define BOOST_MULTI_INDEX_DETAIL_RAW_PTR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost{

namespace multi_index{

namespace detail{



template<typename RawPointer>
inline RawPointer raw_ptr(RawPointer const& p,mpl::true_)
{
return p;
}

template<typename RawPointer,typename Pointer>
inline RawPointer raw_ptr(Pointer const& p,mpl::false_)
{
return p==Pointer(0)?0:&*p;
}

template<typename RawPointer,typename Pointer>
inline RawPointer raw_ptr(Pointer const& p)
{
return raw_ptr<RawPointer>(p,is_same<RawPointer,Pointer>());
}

} 

} 

} 

#endif
