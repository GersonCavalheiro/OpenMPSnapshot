
#ifndef ASIO_ERROR_CODE_HPP
#define ASIO_ERROR_CODE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)
# include <system_error>
#else 
# include <string>
# include "asio/detail/noncopyable.hpp"
# if !defined(ASIO_NO_IOSTREAM)
#  include <iosfwd>
# endif 
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)

typedef std::error_category error_category;

#else 

class error_category : private noncopyable
{
public:
virtual ~error_category()
{
}

virtual const char* name() const = 0;

virtual std::string message(int value) const = 0;

bool operator==(const error_category& rhs) const
{
return this == &rhs;
}

bool operator!=(const error_category& rhs) const
{
return !(*this == rhs);
}
};

#endif 

extern ASIO_DECL const error_category& system_category();

#if defined(ASIO_HAS_STD_SYSTEM_ERROR)

typedef std::error_code error_code;

#else 

class error_code
{
public:
error_code()
: value_(0),
category_(&system_category())
{
}

error_code(int v, const error_category& c)
: value_(v),
category_(&c)
{
}

template <typename ErrorEnum>
error_code(ErrorEnum e)
{
*this = make_error_code(e);
}

void clear()
{
value_ = 0;
category_ = &system_category();
}

void assign(int v, const error_category& c)
{
value_ = v;
category_ = &c;
}

int value() const
{
return value_;
}

const error_category& category() const
{
return *category_;
}

std::string message() const
{
return category_->message(value_);
}

struct unspecified_bool_type_t
{
};

typedef void (*unspecified_bool_type)(unspecified_bool_type_t);

static void unspecified_bool_true(unspecified_bool_type_t) {}

operator unspecified_bool_type() const
{
if (value_ == 0)
return 0;
else
return &error_code::unspecified_bool_true;
}

bool operator!() const
{
return value_ == 0;
}

friend bool operator==(const error_code& e1, const error_code& e2)
{
return e1.value_ == e2.value_ && e1.category_ == e2.category_;
}

friend bool operator!=(const error_code& e1, const error_code& e2)
{
return e1.value_ != e2.value_ || e1.category_ != e2.category_;
}

private:
int value_;

const error_category* category_;
};

# if !defined(ASIO_NO_IOSTREAM)

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
std::basic_ostream<Elem, Traits>& os, const error_code& ec)
{
os << ec.category().name() << ':' << ec.value();
return os;
}

# endif 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/error_code.ipp"
#endif 

#endif 
