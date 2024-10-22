

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_COMPARISON_ADAPTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_DETAIL_COMPARISON_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/call_traits.hpp>

namespace boost {
namespace bimaps {
namespace container_adaptor {
namespace detail {



template < class Compare, class NewType, class Converter >
struct comparison_adaptor
{
typedef NewType first_argument_type;
typedef NewType second_argument_type;
typedef bool result_type;

comparison_adaptor( const Compare & comp, const Converter & conv)
: compare(comp), converter(conv) {}

bool operator()( BOOST_DEDUCED_TYPENAME call_traits<NewType>::param_type x,
BOOST_DEDUCED_TYPENAME call_traits<NewType>::param_type y) const
{
return compare( converter(x), converter(y) );
}

private:
Compare     compare;
Converter   converter;
};

template < class Compare, class NewType, class Converter >
struct compatible_comparison_adaptor
{
typedef NewType first_argument_type;
typedef NewType second_argument_type;
typedef bool result_type;

compatible_comparison_adaptor( const Compare & comp, const Converter & conv)
: compare(comp), converter(conv) {}

template< class CompatibleTypeLeft, class CompatibleTypeRight >
bool operator()( const CompatibleTypeLeft  & x,
const CompatibleTypeRight & y) const
{
return compare( converter(x), converter(y) );
}

private:
Compare     compare;
Converter   converter;
};




template < class Compare, class NewType, class Converter >
struct unary_check_adaptor
{
typedef BOOST_DEDUCED_TYPENAME call_traits<NewType>::param_type argument_type;
typedef bool result_type;

unary_check_adaptor( const Compare & comp, const Converter & conv ) :
compare(comp), converter(conv) {}

bool operator()( BOOST_DEDUCED_TYPENAME call_traits<NewType>::param_type x) const
{
return compare( converter(x) );
}

private:
Compare   compare;
Converter converter;
};

} 
} 
} 
} 


#endif 


