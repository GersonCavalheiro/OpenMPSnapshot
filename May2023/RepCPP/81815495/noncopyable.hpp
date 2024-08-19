
#ifndef ASIO_DETAIL_NONCOPYABLE_HPP
#define ASIO_DETAIL_NONCOPYABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

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

using asio::detail::noncopyable;

} 

#include "asio/detail/pop_options.hpp"

#endif 
