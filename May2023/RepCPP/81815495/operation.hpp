
#ifndef ASIO_DETAIL_OPERATION_HPP
#define ASIO_DETAIL_OPERATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_operation.hpp"
#else
# include "asio/detail/scheduler_operation.hpp"
#endif

namespace asio {
namespace detail {

#if defined(ASIO_HAS_IOCP)
typedef win_iocp_operation operation;
#else
typedef scheduler_operation operation;
#endif

} 
} 

#endif 
