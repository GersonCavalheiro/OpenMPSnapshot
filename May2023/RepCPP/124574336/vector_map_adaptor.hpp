

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_VECTOR_MAP_ADAPTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_VECTOR_MAP_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/list.hpp>
#include <boost/mpl/push_front.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/bimap/container_adaptor/vector_adaptor.hpp>
#include <boost/bimap/container_adaptor/detail/identity_converters.hpp>
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

class FunctorsFromDerivedClasses = mpl::vector<>
>
class vector_map_adaptor :

public vector_adaptor
<
Base,
Iterator, ConstIterator, ReverseIterator, ConstReverseIterator,
IteratorToBaseConverter, IteratorFromBaseConverter,
ReverseIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
FunctorsFromDerivedClasses
>
{
typedef vector_adaptor
<
Base,
Iterator, ConstIterator, ReverseIterator, ConstReverseIterator,
IteratorToBaseConverter, IteratorFromBaseConverter,
ReverseIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
FunctorsFromDerivedClasses

> base_;


public:

typedef BOOST_DEDUCED_TYPENAME Iterator::value_type::first_type  key_type;
typedef BOOST_DEDUCED_TYPENAME Iterator::value_type::second_type data_type;
typedef data_type mapped_type;


public:

vector_map_adaptor() {}

explicit vector_map_adaptor(Base & c) :
base_(c) {}

protected:

typedef vector_map_adaptor vector_map_adaptor_;

};


} 
} 
} 


#endif 

