

#ifndef BOOST_FLYWEIGHT_ASSOC_CONTAINER_FACTORY_HPP
#define BOOST_FLYWEIGHT_ASSOC_CONTAINER_FACTORY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/assoc_container_factory_fwd.hpp>
#include <boost/flyweight/detail/is_placeholder_expr.hpp>
#include <boost/flyweight/detail/nested_xxx_if_not_ph.hpp>
#include <boost/flyweight/factory_tag.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/aux_/lambda_support.hpp>
#include <boost/mpl/if.hpp>

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#include <utility>
#endif

namespace boost{namespace flyweights{namespace detail{
BOOST_FLYWEIGHT_NESTED_XXX_IF_NOT_PLACEHOLDER_EXPRESSION_DEF(iterator)
BOOST_FLYWEIGHT_NESTED_XXX_IF_NOT_PLACEHOLDER_EXPRESSION_DEF(value_type)
}}} 



namespace boost{

namespace flyweights{

template<typename Container>
class assoc_container_factory_class:public factory_marker
{
public:


typedef typename detail::nested_iterator_if_not_placeholder_expression<
Container
>::type                                handle_type;
typedef typename detail::nested_value_type_if_not_placeholder_expression<
Container
>::type                                entry_type;

handle_type insert(const entry_type& x)
{
return cont.insert(x).first;
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
handle_type insert(entry_type&& x)
{
return cont.insert(std::move(x)).first;
}
#endif

void erase(handle_type h)
{
cont.erase(h);
}

static const entry_type& entry(handle_type h){return *h;}

private:


typedef typename mpl::if_<
detail::is_placeholder_expression<Container>,
int,
Container
>::type container_type;
container_type cont;

public:
typedef assoc_container_factory_class type;
BOOST_MPL_AUX_LAMBDA_SUPPORT(1,assoc_container_factory_class,(Container))
};



template<
typename ContainerSpecifier
BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION_DEF
>
struct assoc_container_factory:factory_marker
{
template<typename Entry,typename Key>
struct apply
{
typedef assoc_container_factory_class<
typename mpl::apply2<ContainerSpecifier,Entry,Key>::type
> type;
};
};

}  

} 

#endif
