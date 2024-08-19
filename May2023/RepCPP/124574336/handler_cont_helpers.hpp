
#ifndef BOOST_ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP
#define BOOST_ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/handler_continuation_hook.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost_asio_handler_cont_helpers {

template <typename Context>
inline bool is_continuation(Context& context)
{
#if !defined(BOOST_ASIO_HAS_HANDLER_HOOKS)
return false;
#else
using boost::asio::asio_handler_is_continuation;
return asio_handler_is_continuation(
boost::asio::detail::addressof(context));
#endif
}

} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
