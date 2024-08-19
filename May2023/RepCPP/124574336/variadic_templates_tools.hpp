
#ifndef BOOST_INTERPROCESS_DETAIL_VARIADIC_TEMPLATES_TOOLS_HPP
#define BOOST_INTERPROCESS_DETAIL_VARIADIC_TEMPLATES_TOOLS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/variadic_templates_tools.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

using boost::container::dtl::tuple;
using boost::container::dtl::build_number_seq;
using boost::container::dtl::index_tuple;
using boost::container::dtl::get;

}}}   

#endif   
