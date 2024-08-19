

#ifndef BOOST_MULTI_INDEX_DETAIL_PROMOTES_ARG_HPP
#define BOOST_MULTI_INDEX_DETAIL_PROMOTES_ARG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>



#if BOOST_WORKAROUND(BOOST_MSVC,<1400)

namespace boost{

namespace multi_index{

namespace detail{

template<typename F,typename Arg1,typename Arg2>
struct promotes_1st_arg:mpl::false_{};

template<typename F,typename Arg1,typename Arg2>
struct promotes_2nd_arg:mpl::false_{};

} 

} 

} 

#else

#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/not.hpp>
#include <boost/multi_index/detail/is_transparent.hpp>
#include <boost/type_traits/is_convertible.hpp>

namespace boost{

namespace multi_index{

namespace detail{

template<typename F,typename Arg1,typename Arg2>
struct promotes_1st_arg:
mpl::and_<
mpl::not_<is_transparent<F,Arg1,Arg2> >,
is_convertible<const Arg1,Arg2>,
is_transparent<F,Arg2,Arg2>
>
{};

template<typename F,typename Arg1,typename Arg2>
struct promotes_2nd_arg:
mpl::and_<
mpl::not_<is_transparent<F,Arg1,Arg2> >,
is_convertible<const Arg2,Arg1>,
is_transparent<F,Arg1,Arg1>
>
{};

} 

} 

} 

#endif
#endif
