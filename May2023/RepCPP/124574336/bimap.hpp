










#ifndef BOOST_BIMAP_BIMAP_HPP
#define BOOST_BIMAP_BIMAP_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/bimap/detail/user_interface_config.hpp>
#include <boost/mpl/aux_/na.hpp>

#ifndef BOOST_BIMAP_DISABLE_SERIALIZATION
#include <boost/serialization/nvp.hpp>
#endif 

#include <boost/bimap/detail/bimap_core.hpp>
#include <boost/bimap/detail/map_view_base.hpp>
#include <boost/bimap/detail/modifier_adaptor.hpp>
#include <boost/bimap/relation/support/data_extractor.hpp>
#include <boost/bimap/relation/support/member_with_tag.hpp>

#include <boost/bimap/support/map_type_by.hpp>
#include <boost/bimap/support/map_by.hpp>
#include <boost/bimap/support/iterator_type_by.hpp>


namespace boost {


namespace bimaps {




template
<
class KeyTypeA, class KeyTypeB,
class AP1 = ::boost::mpl::na,
class AP2 = ::boost::mpl::na,
class AP3 = ::boost::mpl::na
>
class bimap
:

public ::boost::bimaps::detail::bimap_core<KeyTypeA,KeyTypeB,AP1,AP2,AP3>,


public ::boost::bimaps::detail::bimap_core<KeyTypeA,KeyTypeB,AP1,AP2,AP3>
::relation_set,


public ::boost::bimaps::detail:: left_map_view_extra_typedefs<
BOOST_DEDUCED_TYPENAME ::boost::bimaps::detail::left_map_view_type<
::boost::bimaps::detail::bimap_core<KeyTypeA,KeyTypeB,AP1,AP2,AP3>
>::type
>,
public ::boost::bimaps::detail::right_map_view_extra_typedefs< 
BOOST_DEDUCED_TYPENAME ::boost::bimaps::detail::right_map_view_type<
::boost::bimaps::detail::bimap_core<KeyTypeA,KeyTypeB,AP1,AP2,AP3>
>::type
>
{
typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::detail::
bimap_core<KeyTypeA,KeyTypeB,AP1,AP2,AP3> base_;

BOOST_DEDUCED_TYPENAME base_::core_type core;

public:




typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::detail::
left_map_view_type<base_>::type  left_map;
typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::detail::
right_map_view_type<base_>::type right_map;

typedef BOOST_DEDUCED_TYPENAME
left_map::iterator        left_iterator;
typedef BOOST_DEDUCED_TYPENAME
left_map::const_iterator  left_const_iterator;

typedef BOOST_DEDUCED_TYPENAME
right_map::iterator       right_iterator;
typedef BOOST_DEDUCED_TYPENAME
right_map::const_iterator right_const_iterator;

typedef BOOST_DEDUCED_TYPENAME
left_map::reference       left_reference;
typedef BOOST_DEDUCED_TYPENAME
left_map::const_reference left_const_reference;

typedef BOOST_DEDUCED_TYPENAME
right_map::reference       right_reference;
typedef BOOST_DEDUCED_TYPENAME
right_map::const_reference right_const_reference;

typedef BOOST_DEDUCED_TYPENAME base_::relation::info_type info_type;

typedef BOOST_DEDUCED_TYPENAME base_::core_type::allocator_type allocator_type; 

left_map  left;

right_map right;

typedef BOOST_DEDUCED_TYPENAME base_::logic_relation_set_tag 
logic_relation_set_tag;
typedef BOOST_DEDUCED_TYPENAME base_::logic_left_tag logic_left_tag;
typedef BOOST_DEDUCED_TYPENAME base_::logic_right_tag logic_right_tag;
typedef BOOST_DEDUCED_TYPENAME base_::core_type::ctor_args_list 
ctor_args_list;

bimap(const allocator_type& al = allocator_type()) :

base_::relation_set(
::boost::multi_index::get<
logic_relation_set_tag
>(core)
),

core(al),

left (
::boost::multi_index::get<
logic_left_tag
>(core)
),
right (
::boost::multi_index::get<
logic_right_tag
>(core)
)

{}

template< class InputIterator >
bimap(InputIterator first,InputIterator last,
const allocator_type& al = allocator_type()) :

base_::relation_set(
::boost::multi_index::get<
BOOST_DEDUCED_TYPENAME base_::logic_relation_set_tag>(core)
),

core(first,last,ctor_args_list(),al),

left (
::boost::multi_index::get<
BOOST_DEDUCED_TYPENAME base_::logic_left_tag>(core)
),
right (
::boost::multi_index::get<
BOOST_DEDUCED_TYPENAME base_::logic_right_tag>(core)
)

{}

bimap(const bimap& x) :

base_::relation_set(
::boost::multi_index::get<
BOOST_DEDUCED_TYPENAME base_::logic_relation_set_tag>(core)
),

core(x.core),

left (
::boost::multi_index::get<
BOOST_DEDUCED_TYPENAME base_::logic_left_tag>(core)
),
right (
::boost::multi_index::get<
BOOST_DEDUCED_TYPENAME base_::logic_right_tag>(core)
)

{}

bimap& operator=(const bimap& x)
{
core = x.core;
return *this;
}


template< class IteratorType >
left_iterator project_left(IteratorType iter)
{
return core.template project<
BOOST_DEDUCED_TYPENAME base_::logic_left_tag>(iter.base());
}

template< class IteratorType >
left_const_iterator project_left(IteratorType iter) const
{
return core.template project<
BOOST_DEDUCED_TYPENAME base_::logic_left_tag>(iter.base());
}

template< class IteratorType >
right_iterator project_right(IteratorType iter)
{
return core.template project<
BOOST_DEDUCED_TYPENAME base_::logic_right_tag>(iter.base());
}

template< class IteratorType >
right_const_iterator project_right(IteratorType iter) const
{
return core.template project<
BOOST_DEDUCED_TYPENAME base_::logic_right_tag>(iter.base());
}

template< class IteratorType >
BOOST_DEDUCED_TYPENAME base_::relation_set::iterator
project_up(IteratorType iter)
{
return core.template project<
BOOST_DEDUCED_TYPENAME base_::logic_relation_set_tag>(iter.base());
}

template< class IteratorType >
BOOST_DEDUCED_TYPENAME base_::relation_set::const_iterator
project_up(IteratorType iter) const
{
return core.template project<
BOOST_DEDUCED_TYPENAME base_::logic_relation_set_tag>(iter.base());
}


template< class Tag, class IteratorType >
BOOST_DEDUCED_TYPENAME ::boost::bimaps::support::
iterator_type_by<Tag,bimap>::type
project(IteratorType iter)
{
return core.template project<Tag>(iter.base());
}

template< class Tag, class IteratorType >
BOOST_DEDUCED_TYPENAME ::boost::bimaps::support::
const_iterator_type_by<Tag,bimap>::type
project(IteratorType iter) const
{
return core.template project<Tag>(iter.base());
}

template< class Tag >
struct map_by :
public ::boost::bimaps::support::map_type_by<Tag,bimap>::type
{
typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::support::
map_type_by<Tag,bimap>::type type;

private: map_by() {}
};

template< class Tag >
BOOST_DEDUCED_TYPENAME ::boost::bimaps::support::
map_type_by<Tag,bimap>::type &by()
{
return ::boost::bimaps::support::map_by<Tag>(*this);
}

template< class Tag >
const BOOST_DEDUCED_TYPENAME ::boost::bimaps::support::
map_type_by<Tag,bimap>::type &by() const
{
return ::boost::bimaps::support::map_by<Tag>(*this);
}


#ifndef BOOST_BIMAP_DISABLE_SERIALIZATION


private:

friend class boost::serialization::access;

template<class Archive>
void serialize(Archive & ar, const unsigned int)
{
ar & serialization::make_nvp("mi_core",core);
}

#endif 
};

} 
} 











#include <boost/bimap/tags/tagged.hpp>
#include <boost/bimap/relation/member_at.hpp>
#include <boost/multi_index/detail/unbounded.hpp>

namespace boost {
namespace bimaps {

using ::boost::bimaps::tags::tagged;

namespace member_at = ::boost::bimaps::relation::member_at;

using ::boost::multi_index::unbounded;

} 
} 


#endif 
