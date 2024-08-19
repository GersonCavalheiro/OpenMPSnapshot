

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_UNORDERED_SET_ADAPTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_UNORDERED_SET_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/unordered_associative_container_adaptor.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/mpl/vector.hpp>

namespace boost {
namespace bimaps {
namespace container_adaptor {


template
<
class Base,

class Iterator,
class ConstIterator,
class LocalIterator,
class ConstLocalIterator,

class IteratorToBaseConverter        = ::boost::mpl::na,
class IteratorFromBaseConverter      = ::boost::mpl::na,
class LocalIteratorFromBaseConverter = ::boost::mpl::na,
class ValueToBaseConverter           = ::boost::mpl::na,
class ValueFromBaseConverter         = ::boost::mpl::na,
class KeyToBaseConverter             = ::boost::mpl::na,

class FunctorsFromDerivedClasses = mpl::vector<>
>
class unordered_set_adaptor :

public ::boost::bimaps::container_adaptor::
unordered_associative_container_adaptor
<
Base,
Iterator, ConstIterator, LocalIterator, ConstLocalIterator,
BOOST_DEDUCED_TYPENAME Iterator::value_type,
IteratorToBaseConverter, IteratorFromBaseConverter,
LocalIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
KeyToBaseConverter,
FunctorsFromDerivedClasses
>
{
typedef ::boost::bimaps::container_adaptor::
unordered_associative_container_adaptor
<
Base,
Iterator, ConstIterator, LocalIterator, ConstLocalIterator,
BOOST_DEDUCED_TYPENAME Iterator::value_type,
IteratorToBaseConverter, IteratorFromBaseConverter,
LocalIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
KeyToBaseConverter,
FunctorsFromDerivedClasses

> base_;


public:

explicit unordered_set_adaptor(Base & c) :
base_(c) {}

protected:

typedef unordered_set_adaptor unordered_set_adaptor_;

};


} 
} 
} 


#endif 

