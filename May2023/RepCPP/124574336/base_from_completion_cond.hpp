
#ifndef BOOST_ASIO_DETAIL_BASE_FROM_COMPLETION_COND_HPP
#define BOOST_ASIO_DETAIL_BASE_FROM_COMPLETION_COND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/completion_condition.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename CompletionCondition>
class base_from_completion_cond
{
protected:
explicit base_from_completion_cond(CompletionCondition& completion_condition)
: completion_condition_(
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition))
{
}

std::size_t check_for_completion(
const boost::system::error_code& ec,
std::size_t total_transferred)
{
return detail::adapt_completion_condition_result(
completion_condition_(ec, total_transferred));
}

private:
CompletionCondition completion_condition_;
};

template <>
class base_from_completion_cond<transfer_all_t>
{
protected:
explicit base_from_completion_cond(transfer_all_t)
{
}

static std::size_t check_for_completion(
const boost::system::error_code& ec,
std::size_t total_transferred)
{
return transfer_all_t()(ec, total_transferred);
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
