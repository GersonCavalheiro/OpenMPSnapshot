
#ifndef ASIO_ASSOCIATOR_HPP
#define ASIO_ASSOCIATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <template <typename, typename> class Associator,
typename T, typename DefaultCandidate>
struct associator
{
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
