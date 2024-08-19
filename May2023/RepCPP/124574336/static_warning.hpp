#ifndef BOOST_SERIALIZATION_STATIC_WARNING_HPP
#define BOOST_SERIALIZATION_STATIC_WARNING_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>





#include <boost/mpl/bool.hpp>
#include <boost/mpl/print.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/bool_fwd.hpp>
#include <boost/static_assert.hpp>

namespace boost {
namespace serialization {

template<int L>
struct BOOST_SERIALIZATION_STATIC_WARNING_LINE{};

template<bool B, int L>
struct static_warning_test{
typename boost::mpl::eval_if_c<
B,
boost::mpl::true_,
typename boost::mpl::identity<
boost::mpl::print<
BOOST_SERIALIZATION_STATIC_WARNING_LINE<L>
>
>
>::type type;
};

template<int i>
struct BOOST_SERIALIZATION_SS {};

} 
} 

#define BOOST_SERIALIZATION_BSW(B, L) \
typedef boost::serialization::BOOST_SERIALIZATION_SS< \
sizeof( boost::serialization::static_warning_test< B, L > ) \
> BOOST_JOIN(STATIC_WARNING_LINE, L) BOOST_ATTRIBUTE_UNUSED;
#define BOOST_STATIC_WARNING(B) BOOST_SERIALIZATION_BSW(B, __LINE__)

#endif 
