

#ifndef BOOST_MULTI_INDEX_DETAIL_ADL_SWAP_HPP
#define BOOST_MULTI_INDEX_DETAIL_ADL_SWAP_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>

namespace boost{

namespace multi_index{

namespace detail{

template<typename T>
void adl_swap(T& x,T& y)
{

#if !defined(BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL)
using std::swap;
swap(x,y);
#else
std::swap(x,y);
#endif

}

} 

} 

} 

#endif
