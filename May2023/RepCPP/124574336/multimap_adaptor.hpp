

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_MULTIMAP_ADAPTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_MULTIMAP_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/ordered_associative_container_adaptor.hpp>
#include <boost/bimap/container_adaptor/detail/non_unique_container_helper.hpp>
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
class multimap_adaptor :

public ::boost::bimaps::container_adaptor::
ordered_associative_container_adaptor
<
Base,
Iterator, ConstIterator, ReverseIterator, ConstReverseIterator,
BOOST_DEDUCED_TYPENAME Iterator::value_type::first_type,
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
BOOST_DEDUCED_TYPENAME Iterator::value_type::first_type,
IteratorToBaseConverter, IteratorFromBaseConverter,
ReverseIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
KeyToBaseConverter,
FunctorsFromDerivedClasses

> base_;


public:

typedef BOOST_DEDUCED_TYPENAME Iterator::value_type::second_type data_type;
typedef data_type mapped_type;


public:

explicit multimap_adaptor(Base & c) :
base_(c) {}

protected:

typedef multimap_adaptor multimap_adaptor_;

public:

BOOST_BIMAP_NON_UNIQUE_CONTAINER_ADAPTOR_INSERT_FUNCTIONS
};


} 
} 
} 


#endif 


