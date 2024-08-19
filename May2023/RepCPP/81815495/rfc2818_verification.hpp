
#ifndef ASIO_SSL_RFC2818_VERIFICATION_HPP
#define ASIO_SSL_RFC2818_VERIFICATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_DEPRECATED)

#include <string>
#include "asio/ssl/detail/openssl_types.hpp"
#include "asio/ssl/verify_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {


class rfc2818_verification
{
public:
typedef bool result_type;

explicit rfc2818_verification(const std::string& host)
: host_(host)
{
}

ASIO_DECL bool operator()(bool preverified, verify_context& ctx) const;

private:
ASIO_DECL static bool match_pattern(const char* pattern,
std::size_t pattern_length, const char* host);

std::string host_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/impl/rfc2818_verification.ipp"
#endif 

#endif 

#endif 
