

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_SET_ADAPTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_SET_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/ordered_associative_container_adaptor.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/aux_/na.hpp>

namespace boost {
namespace bimaps {
namespace container_adaptor {


template
<
class Base,

class Iterator,
class ConstIterator,
class ReverseIterator,
class ConstReverseIterator,

class IteratorToBaseConverter          = ::boost::mpl::na,
class IteratorFromBaseConverter        = ::boost::mpl::na,
class ReverseIteratorFromBaseConverter = ::boost::mpl::na,
class ValueToBaseConverter             = ::boost::mpl::na,
class ValueFromBaseConverter           = ::boost::mpl::na,
class KeyToBaseConverter               = ::boost::mpl::na,

class FunctorsFromDerivedClasses = mpl::vector<>
>
class set_adaptor :

public ::boost::bimaps::container_adaptor::
ordered_associative_container_adaptor
<
Base,
Iterator, ConstIterator, ReverseIterator, ConstReverseIterator,
BOOST_DEDUCED_TYPENAME Iterator::value_type,
IteratorToBaseConverter, IteratorFromBaseConverter,
ReverseIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
KeyToBaseConverter,
FunctorsFromDerivedClasses
>
{

typedef ::boost::bimaps::container_adaptor::
ordered_associative_container_adaptor
<
Base,
Iterator, ConstIterator, ReverseIterator, ConstReverseIterator,
BOOST_DEDUCED_TYPENAME Iterator::value_type,
IteratorToBaseConverter, IteratorFromBaseConverter,
ReverseIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
KeyToBaseConverter,
FunctorsFromDerivedClasses

> base_;


public:

explicit set_adaptor(Base & c) :
base_(c) {}

protected:

typedef set_adaptor set_adaptor_;

};


} 
} 
} 


#endif 


