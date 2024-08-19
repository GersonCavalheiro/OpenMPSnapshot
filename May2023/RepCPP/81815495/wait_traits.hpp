
#ifndef ASIO_WAIT_TRAITS_HPP
#define ASIO_WAIT_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename Clock>
struct wait_traits
{

static typename Clock::duration to_wait_duration(
const typename Clock::duration& d)
{
return d;
}


static typename Clock::duration to_wait_duration(
const typename Clock::time_point& t)
{
typename Clock::time_point now = Clock::now();
if (now + (Clock::duration::max)() < t)
return (Clock::duration::max)();
if (now + (Clock::duration::min)() > t)
return (Clock::duration::min)();
return t - now;
}
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
