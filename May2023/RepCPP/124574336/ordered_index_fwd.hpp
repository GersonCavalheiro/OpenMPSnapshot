

#ifndef BOOST_MULTI_INDEX_ORDERED_INDEX_FWD_HPP
#define BOOST_MULTI_INDEX_ORDERED_INDEX_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/multi_index/detail/ord_index_args.hpp>
#include <boost/multi_index/detail/ord_index_impl_fwd.hpp>

namespace boost{

namespace multi_index{



template<typename Arg1,typename Arg2=mpl::na,typename Arg3=mpl::na>
struct ordered_unique;

template<typename Arg1,typename Arg2=mpl::na,typename Arg3=mpl::na>
struct ordered_non_unique;

} 

} 

#endif
