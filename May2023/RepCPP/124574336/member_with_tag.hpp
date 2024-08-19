

#ifndef BOOST_BIMAP_RELATION_SUPPORT_MEMBER_WITH_TAG_HPP
#define BOOST_BIMAP_RELATION_SUPPORT_MEMBER_WITH_TAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/member_at.hpp>
#include <boost/bimap/detail/debug/static_error.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/and.hpp>



#ifndef BOOST_BIMAP_DOXYGEN_WILL_NOT_PROCESS_THE_FOLLOWING_LINES

namespace boost {
namespace bimaps {
namespace relation {
namespace support {

template
<
class Tag,
class Relation,
class Enable = void
>
struct member_with_tag
{
BOOST_BIMAP_STATIC_ERROR( MEMBER_WITH_TAG_FAILURE, (Relation,Tag) );
};

template< class Relation >
struct member_with_tag
<
member_at::left, Relation, void
>
{
typedef member_at::left type;
};

template< class Relation >
struct member_with_tag
<
member_at::right, Relation, void
>
{
typedef member_at::right type;
};

template< class Relation >
struct member_with_tag
<
member_at::info, Relation, void
>
{
typedef member_at::info type;
};


template< class Tag, class Relation >
struct member_with_tag
<
Tag, Relation,
BOOST_DEDUCED_TYPENAME enable_if
<
mpl::and_
<
mpl::not_< is_same<Tag,member_at::left> >,
is_same
<
Tag,
BOOST_DEDUCED_TYPENAME Relation::left_tag
>
>

>::type
>
{
typedef member_at::left type;
};

template< class Tag, class Relation >
struct member_with_tag
<
Tag,
Relation,
BOOST_DEDUCED_TYPENAME enable_if
<
mpl::and_
<
mpl::not_< is_same<Tag,member_at::right> >,
is_same
<
Tag,
BOOST_DEDUCED_TYPENAME Relation::right_tag
>
>

>::type
>
{
typedef member_at::right type;
};

template< class Tag, class Relation >
struct member_with_tag
<
Tag, Relation,
BOOST_DEDUCED_TYPENAME enable_if
<
mpl::and_
<
mpl::not_< is_same<Tag,member_at::info> >,
is_same
<
Tag,
BOOST_DEDUCED_TYPENAME Relation::info_tag
>
>

>::type
>
{
typedef member_at::info type;
};

} 
} 
} 
} 

#endif 

#endif 


