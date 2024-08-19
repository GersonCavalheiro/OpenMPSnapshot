

#ifndef BOOST_BIMAP_DETAIL_MODIFIER_ADAPTOR_HPP
#define BOOST_BIMAP_DETAIL_MODIFIER_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

namespace boost {
namespace bimaps {
namespace detail {


template
<
class Modifier,
class NewArgument,
class FirstExtractor,
class SecondExtractor 
>
struct relation_modifier_adaptor :
Modifier,
FirstExtractor,
SecondExtractor
{
typedef NewArgument argument_type;
typedef void result_type;

relation_modifier_adaptor( const Modifier & m ) : Modifier(m) {}
relation_modifier_adaptor( const Modifier & m,
const FirstExtractor & fe,
const SecondExtractor & se ) :
Modifier(m), FirstExtractor(fe), SecondExtractor(se) {}

void operator()( NewArgument & x ) const
{
Modifier::operator()(
FirstExtractor ::operator()( x ),
SecondExtractor::operator()( x )
);
}
};


template
<
class Modifier,
class NewArgument,
class Extractor
>
struct unary_modifier_adaptor :
Modifier,
Extractor
{
typedef NewArgument argument_type;
typedef void result_type;

unary_modifier_adaptor( const Modifier & m ) : Modifier(m) {}
unary_modifier_adaptor( const Modifier & m,
const Extractor & fe) :
Modifier(m), Extractor(fe) {}

void operator()( NewArgument & x ) const
{
Modifier::operator()( Extractor::operator()( x ) );
}
};


} 
} 
} 


#endif 
