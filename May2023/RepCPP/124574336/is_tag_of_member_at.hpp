

#ifndef BOOST_BIMAP_RELATION_SUPPORT_IS_TAG_OF_MEMBER_AT_HPP
#define BOOST_BIMAP_RELATION_SUPPORT_IS_TAG_OF_MEMBER_AT_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/bimap/relation/member_at.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/bimap/relation/support/member_with_tag.hpp>







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
struct is_tag_of_member_at_left :
::boost::mpl::false_ {};

template< class Tag, class Relation >
struct is_tag_of_member_at_left
<
Tag, Relation,
BOOST_DEDUCED_TYPENAME enable_if
<
is_same
<
BOOST_DEDUCED_TYPENAME member_with_tag<Tag,Relation>::type,
member_at::left
>

>::type
> :
::boost::mpl::true_ {};


template
<
class Tag,
class Relation,
class Enable = void
>
struct is_tag_of_member_at_right :
::boost::mpl::false_ {};

template< class Tag, class Relation >
struct is_tag_of_member_at_right
<
Tag, Relation,
BOOST_DEDUCED_TYPENAME enable_if
<
is_same
<
BOOST_DEDUCED_TYPENAME member_with_tag<Tag,Relation>::type,
member_at::right
>

>::type
> :
::boost::mpl::true_ {};



template
<
class Tag,
class Relation,
class Enable = void
>
struct is_tag_of_member_at_info :
::boost::mpl::false_ {};

template< class Tag, class Relation >
struct is_tag_of_member_at_info
<
Tag, Relation,
BOOST_DEDUCED_TYPENAME enable_if
<
is_same
<
BOOST_DEDUCED_TYPENAME member_with_tag<Tag,Relation>::type,
member_at::info
>

>::type
> :
::boost::mpl::true_ {};

} 
} 
} 
} 

#endif 

#endif 


