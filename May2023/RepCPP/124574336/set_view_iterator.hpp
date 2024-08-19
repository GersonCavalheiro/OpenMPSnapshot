

#ifndef BOOST_BIMAP_DETAIL_SET_VIEW_ITERATOR_HPP
#define BOOST_BIMAP_DETAIL_SET_VIEW_ITERATOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>


#ifndef BOOST_BIMAP_DISABLE_SERIALIZATION 
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#endif 

#include <boost/iterator/detail/enable_if.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/bimap/relation/support/get_pair_functor.hpp>

namespace boost {
namespace bimaps {
namespace detail {




#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

template< class CoreIterator > struct set_view_iterator;

template< class CoreIterator >
struct set_view_iterator_base
{
typedef iterator_adaptor
<
set_view_iterator< CoreIterator >,
CoreIterator,
BOOST_DEDUCED_TYPENAME CoreIterator::value_type::above_view

> type;
};

#endif 

template< class CoreIterator >
struct set_view_iterator : public set_view_iterator_base<CoreIterator>::type
{
typedef BOOST_DEDUCED_TYPENAME set_view_iterator_base<CoreIterator>::type base_;

public:

set_view_iterator() {}

set_view_iterator(CoreIterator const& iter)
: base_(iter) {}

set_view_iterator(set_view_iterator const & iter)
: base_(iter.base()) {}

typename base_::reference dereference() const
{
return const_cast<
BOOST_DEDUCED_TYPENAME base_::base_type::value_type*>(
&(*this->base())
)->get_view();
}

private:

friend class iterator_core_access;

#ifndef BOOST_BIMAP_DISABLE_SERIALIZATION


BOOST_SERIALIZATION_SPLIT_MEMBER()

friend class ::boost::serialization::access;

template< class Archive >
void save(Archive & ar, const unsigned int) const
{
ar << ::boost::serialization::make_nvp("mi_iterator",this->base());
}

template< class Archive >
void load(Archive & ar, const unsigned int)
{
CoreIterator iter;
ar >> ::boost::serialization::make_nvp("mi_iterator",iter);
this->base_reference() = iter;
}

#endif 
};

#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

template< class CoreIterator > struct const_set_view_iterator;

template< class CoreIterator >
struct const_set_view_iterator_base
{
typedef iterator_adaptor
<
const_set_view_iterator< CoreIterator >,
CoreIterator,
const BOOST_DEDUCED_TYPENAME CoreIterator::value_type::above_view

> type;
};

#endif 




template< class CoreIterator >
struct const_set_view_iterator : public const_set_view_iterator_base<CoreIterator>::type
{
typedef BOOST_DEDUCED_TYPENAME const_set_view_iterator_base<CoreIterator>::type base_;

public:

const_set_view_iterator() {}

const_set_view_iterator(CoreIterator const& iter)
: base_(iter) {}

const_set_view_iterator(const_set_view_iterator const & iter)
: base_(iter.base()) {}

const_set_view_iterator(set_view_iterator<CoreIterator> i)
: base_(i.base()) {}

BOOST_DEDUCED_TYPENAME base_::reference dereference() const
{
return this->base()->get_view();
}

private:

friend class iterator_core_access;

#ifndef BOOST_BIMAP_DISABLE_SERIALIZATION


BOOST_SERIALIZATION_SPLIT_MEMBER()

friend class ::boost::serialization::access;

template< class Archive >
void save(Archive & ar, const unsigned int) const
{
ar << ::boost::serialization::make_nvp("mi_iterator",this->base());
}

template< class Archive >
void load(Archive & ar, const unsigned int)
{
CoreIterator iter;
ar >> ::boost::serialization::make_nvp("mi_iterator",iter);
this->base_reference() = iter;
}

#endif 
};


} 
} 
} 

#endif 


