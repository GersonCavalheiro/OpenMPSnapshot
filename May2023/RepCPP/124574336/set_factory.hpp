

#ifndef BOOST_FLYWEIGHT_SET_FACTORY_HPP
#define BOOST_FLYWEIGHT_SET_FACTORY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/allocator_utilities.hpp>
#include <boost/flyweight/assoc_container_factory.hpp>
#include <boost/flyweight/factory_tag.hpp>
#include <boost/flyweight/set_factory_fwd.hpp>
#include <boost/mpl/aux_/lambda_support.hpp>
#include <boost/mpl/if.hpp>
#include <set>



namespace boost{

namespace flyweights{

template<
typename Entry,typename Key,
typename Compare,typename Allocator
>
class set_factory_class:
public assoc_container_factory_class<
std::set<
Entry,
typename boost::mpl::if_<
mpl::is_na<Compare>,
std::less<Key>,
Compare
>::type,
typename boost::mpl::if_<
mpl::is_na<Allocator>,
std::allocator<Entry>,
Allocator
>::type
>
>
{
public:
typedef set_factory_class type;
BOOST_MPL_AUX_LAMBDA_SUPPORT(
4,set_factory_class,(Entry,Key,Compare,Allocator))
};



template<
typename Compare,typename Allocator
BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION_DEF
>
struct set_factory:factory_marker
{
template<typename Entry,typename Key>
struct apply:
mpl::apply2<
set_factory_class<
boost::mpl::_1,boost::mpl::_2,Compare,Allocator
>,
Entry,Key
>
{};
};

} 

} 

#endif
