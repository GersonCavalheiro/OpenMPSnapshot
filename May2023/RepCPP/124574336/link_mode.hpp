
#ifndef BOOST_INTRUSIVE_LINK_MODE_HPP
#define BOOST_INTRUSIVE_LINK_MODE_HPP

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

enum link_mode_type{
normal_link,

safe_link,

auto_unlink
};

#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED

template <link_mode_type link_mode>
struct is_safe_autounlink
{
static const bool value =
(int)link_mode == (int)auto_unlink   ||
(int)link_mode == (int)safe_link;
};

#endif   

} 
} 

#endif 
