
#ifndef ASIO_SYSTEM_ERROR_HPP
#define ASIO_SYSTEM_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)
# include <system_error>
#else 
# include <cerrno>
# include <exception>
# include <string>
# include "asio/error_code.hpp"
# include "asio/detail/scoped_ptr.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)

typedef std::system_error system_error;

#else 

class system_error
: public std::exception
{
public:
system_error(const error_code& ec)
: code_(ec),
context_()
{
}

system_error(const error_code& ec, const std::string& context)
: code_(ec),
context_(context)
{
}

system_error(const system_error& other)
: std::exception(other),
code_(other.code_),
context_(other.context_),
what_()
{
}

virtual ~system_error() throw ()
{
}

system_error& operator=(const system_error& e)
{
context_ = e.context_;
code_ = e.code_;
what_.reset();
return *this;
}

virtual const char* what() const throw ()
{
#if !defined(ASIO_NO_EXCEPTIONS)
try
#endif 
{
if (!what_.get())
{
std::string tmp(context_);
if (tmp.length())
tmp += ": ";
tmp += code_.message();
what_.reset(new std::string(tmp));
}
return what_->c_str();
}
#if !defined(ASIO_NO_EXCEPTIONS)
catch (std::exception&)
{
return "system_error";
}
#endif 
}

error_code code() const
{
return code_;
}

private:
error_code code_;

std::string context_;

mutable asio::detail::scoped_ptr<std::string> what_;
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
