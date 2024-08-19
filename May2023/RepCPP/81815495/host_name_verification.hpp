
#ifndef ASIO_SSL_HOST_NAME_VERIFICATION_HPP
#define ASIO_SSL_HOST_NAME_VERIFICATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include <string>
#include "asio/ssl/detail/openssl_types.hpp"
#include "asio/ssl/verify_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {


class host_name_verification
{
public:
typedef bool result_type;

explicit host_name_verification(const std::string& host)
: host_(host)
{
}

ASIO_DECL bool operator()(bool preverified, verify_context& ctx) const;

private:
std::string host_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/impl/host_name_verification.ipp"
#endif 

#endif 
