
#ifndef ASIO_DETAIL_CSTDDEF_HPP
#define ASIO_DETAIL_CSTDDEF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>

namespace asio {

#if defined(ASIO_HAS_NULLPTR)
using std::nullptr_t;
#else 
struct nullptr_t {};
#endif 

} 

#endif 
