
#ifndef BOOST_INTERPROCESS_DETAIL_CAST_TAGS_HPP
#define BOOST_INTERPROCESS_DETAIL_CAST_TAGS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost { namespace interprocess { namespace ipcdetail {

struct static_cast_tag {};
struct const_cast_tag {};
struct dynamic_cast_tag {};
struct reinterpret_cast_tag {};

}}}  

#endif   
