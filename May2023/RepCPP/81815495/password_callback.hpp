
#ifndef ASIO_SSL_DETAIL_PASSWORD_CALLBACK_HPP
#define ASIO_SSL_DETAIL_PASSWORD_CALLBACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include <cstddef>
#include <string>
#include "asio/ssl/context_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

class password_callback_base
{
public:
virtual ~password_callback_base()
{
}

virtual std::string call(std::size_t size,
context_base::password_purpose purpose) = 0;
};

template <typename PasswordCallback>
class password_callback : public password_callback_base
{
public:
explicit password_callback(PasswordCallback callback)
: callback_(callback)
{
}

virtual std::string call(std::size_t size,
context_base::password_purpose purpose)
{
return callback_(size, purpose);
}

private:
PasswordCallback callback_;
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
