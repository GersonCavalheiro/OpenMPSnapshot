
#ifndef BOOST_ASIO_DETAIL_NONCOPYABLE_HPP
#define BOOST_ASIO_DETAIL_NONCOPYABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class noncopyable
{
protected:
noncopyable() {}
~noncopyable() {}
private:
noncopyable(const noncopyable&);
const noncopyable& operator=(const noncopyable&);
};

} 

using boost::asio::detail::noncopyable;

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
