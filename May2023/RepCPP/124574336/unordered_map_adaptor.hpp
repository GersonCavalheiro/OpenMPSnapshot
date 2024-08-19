

#ifndef BOOST_BIMAP_CONTAINER_ADAPTOR_UNORDERED_MAP_ADAPTOR_HPP
#define BOOST_BIMAP_CONTAINER_ADAPTOR_UNORDERED_MAP_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/container_adaptor/unordered_associative_container_adaptor.hpp>
#include <boost/mpl/aux_/na.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/call_traits.hpp>

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
class unordered_map_adaptor :

public ::boost::bimaps::container_adaptor::
unordered_associative_container_adaptor
<
Base,
Iterator, ConstIterator, LocalIterator, ConstLocalIterator,
BOOST_DEDUCED_TYPENAME Iterator::value_type::first_type,
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
BOOST_DEDUCED_TYPENAME Iterator::value_type::first_type,
IteratorToBaseConverter, IteratorFromBaseConverter,
LocalIteratorFromBaseConverter,
ValueToBaseConverter, ValueFromBaseConverter,
KeyToBaseConverter,
FunctorsFromDerivedClasses

> base_;


public:

typedef BOOST_DEDUCED_TYPENAME Iterator::value_type::second_type data_type;
typedef data_type mapped_type;


public:

explicit unordered_map_adaptor(Base & c) :
base_(c) {}

protected:

typedef unordered_map_adaptor unordered_map_adaptor_;


public:

template< class CompatibleKey >
data_type& operator[](const CompatibleKey & k)
{
return this->base()
[this->template functor<BOOST_DEDUCED_TYPENAME base_::key_to_base>()(k)];
}

template< class CompatibleKey >
data_type& at(const CompatibleKey & k)
{
return this->base().
at(this->template functor<BOOST_DEDUCED_TYPENAME base_::key_to_base>()(k));
}

template< class CompatibleKey >
const data_type& at(const CompatibleKey & k) const
{
return this->base().
at(this->template functor<BOOST_DEDUCED_TYPENAME base_::key_to_base>()(k));
}

};



} 
} 
} 


#endif 

