
#ifndef ASIO_CANCELLATION_TYPE_HPP
#define ASIO_CANCELLATION_TYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

# if defined(GENERATING_DOCUMENTATION)

enum cancellation_type
{
none = 0,

terminal = 1,

partial = 2,

total = 4,

all = 0xFFFFFFFF
};

typedef cancellation_type cancellation_type_t;

#elif defined(ASIO_HAS_ENUM_CLASS)

enum class cancellation_type : unsigned int
{
none = 0,
terminal = 1,
partial = 2,
total = 4,
all = 0xFFFFFFFF
};

typedef cancellation_type cancellation_type_t;

#else 

namespace cancellation_type {

enum cancellation_type_t
{
none = 0,
terminal = 1,
partial = 2,
total = 4,
all = 0xFFFFFFFF
};

} 

typedef cancellation_type::cancellation_type_t cancellation_type_t;

#endif 


inline ASIO_CONSTEXPR bool operator!(cancellation_type_t x)
{
return static_cast<unsigned int>(x) == 0;
}


inline ASIO_CONSTEXPR cancellation_type_t operator&(
cancellation_type_t x, cancellation_type_t y)
{
return static_cast<cancellation_type_t>(
static_cast<unsigned int>(x) & static_cast<unsigned int>(y));
}


inline ASIO_CONSTEXPR cancellation_type_t operator|(
cancellation_type_t x, cancellation_type_t y)
{
return static_cast<cancellation_type_t>(
static_cast<unsigned int>(x) | static_cast<unsigned int>(y));
}


inline ASIO_CONSTEXPR cancellation_type_t operator^(
cancellation_type_t x, cancellation_type_t y)
{
return static_cast<cancellation_type_t>(
static_cast<unsigned int>(x) ^ static_cast<unsigned int>(y));
}


inline ASIO_CONSTEXPR cancellation_type_t operator~(cancellation_type_t x)
{
return static_cast<cancellation_type_t>(~static_cast<unsigned int>(x));
}


inline cancellation_type_t& operator&=(
cancellation_type_t& x, cancellation_type_t y)
{
x = x & y;
return x;
}


inline cancellation_type_t& operator|=(
cancellation_type_t& x, cancellation_type_t y)
{
x = x | y;
return x;
}


inline cancellation_type_t& operator^=(
cancellation_type_t& x, cancellation_type_t y)
{
x = x ^ y;
return x;
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
