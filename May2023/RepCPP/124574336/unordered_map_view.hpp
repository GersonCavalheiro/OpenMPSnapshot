

#ifndef BOOST_BIMAP_VIEWS_UNOREDERED_MAP_VIEW_HPP
#define BOOST_BIMAP_VIEWS_UNOREDERED_MAP_VIEW_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <utility>

#include <boost/bimap/container_adaptor/unordered_map_adaptor.hpp>
#include <boost/bimap/detail/map_view_base.hpp>

namespace boost {
namespace bimaps {
namespace views {




template< class Tag, class BimapType >
class unordered_map_view
:
public BOOST_BIMAP_MAP_VIEW_CONTAINER_ADAPTOR(
unordered_map_adaptor,
Tag,BimapType,
local_map_view_iterator,const_local_map_view_iterator
),

public ::boost::bimaps::detail::map_view_base<
unordered_map_view<Tag,BimapType>,Tag,BimapType >,
public ::boost::bimaps::detail::
unique_map_view_access<
unordered_map_view<Tag,BimapType>, Tag,  BimapType>::type

{
typedef BOOST_BIMAP_MAP_VIEW_CONTAINER_ADAPTOR(
unordered_map_adaptor,
Tag,BimapType,
local_map_view_iterator,const_local_map_view_iterator

) base_;

BOOST_BIMAP_MAP_VIEW_BASE_FRIEND(unordered_map_view,Tag,BimapType)

typedef BOOST_DEDUCED_TYPENAME ::boost::bimaps::detail::
unique_map_view_access<
unordered_map_view<Tag,BimapType>, Tag,  BimapType

>::type unique_map_view_access_;

public:

typedef std::pair<
BOOST_DEDUCED_TYPENAME base_::iterator,
BOOST_DEDUCED_TYPENAME base_::iterator
> range_type;

typedef std::pair<
BOOST_DEDUCED_TYPENAME base_::const_iterator,
BOOST_DEDUCED_TYPENAME base_::const_iterator
> const_range_type;

typedef BOOST_DEDUCED_TYPENAME base_::value_type::info_type info_type;

unordered_map_view(BOOST_DEDUCED_TYPENAME base_::base_type & c)
: base_(c) {}

using unique_map_view_access_::at;
using unique_map_view_access_::operator[];

unordered_map_view & operator=(const unordered_map_view & v) 
{
this->base() = v.base();
return *this;
}


template< class CompatibleKey >
const info_type & info_at(const CompatibleKey& k) const
{
BOOST_DEDUCED_TYPENAME base_::const_iterator iter = this->find(k);
if( iter == this->end() )
{
::boost::throw_exception(
std::out_of_range("bimap<>: invalid key")
);
}
return iter->info;
}

template< class CompatibleKey >
info_type & info_at(const CompatibleKey& k)
{
BOOST_DEDUCED_TYPENAME base_::iterator iter = this->find(k);
if( iter == this->end() )
{
::boost::throw_exception(
std::out_of_range("bimap<>: invalid key")
);
}
return iter->info;
}
};


} 


#define BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,TYPENAME)            \
typedef BOOST_DEDUCED_TYPENAME MAP_VIEW::TYPENAME                             \
BOOST_PP_CAT(SIDE,BOOST_PP_CAT(_,TYPENAME));



#define BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEFS_BODY(MAP_VIEW,SIDE)               \
BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,local_iterator)          \
BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,const_local_iterator)    \
BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,range_type)              \
BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,const_range_type)        \
BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,hasher)                  \
BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF(MAP_VIEW,SIDE,key_equal)


namespace detail {

template< class Tag, class BimapType >
struct left_map_view_extra_typedefs< ::boost::bimaps::views::unordered_map_view<Tag,BimapType> >
{
private: typedef ::boost::bimaps::views::unordered_map_view<Tag,BimapType> map_view_;
public : BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEFS_BODY(map_view_,left)
};

template< class Tag, class BimapType >
struct right_map_view_extra_typedefs< ::boost::bimaps::views::unordered_map_view<Tag,BimapType> >
{
private: typedef ::boost::bimaps::views::unordered_map_view<Tag,BimapType> map_view_;
public : BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEFS_BODY(map_view_,right)
};

} 


#undef BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEF
#undef BOOST_BIMAP_MAP_VIEW_EXTRA_TYPEDEFS_BODY


} 
} 

#endif 


