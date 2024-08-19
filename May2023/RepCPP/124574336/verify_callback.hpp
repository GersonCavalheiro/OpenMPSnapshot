
#ifndef BOOST_ASIO_SSL_DETAIL_VERIFY_CALLBACK_HPP
#define BOOST_ASIO_SSL_DETAIL_VERIFY_CALLBACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/ssl/verify_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ssl {
namespace detail {

class verify_callback_base
{
public:
virtual ~verify_callback_base()
{
}

virtual bool call(bool preverified, verify_context& ctx) = 0;
};

template <typename VerifyCallback>
class verify_callback : public verify_callback_base
{
public:
explicit verify_callback(VerifyCallback callback)
: callback_(callback)
{
}

virtual bool call(bool preverified, verify_context& ctx)
{
return callback_(preverified, ctx);
}

private:
VerifyCallback callback_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
