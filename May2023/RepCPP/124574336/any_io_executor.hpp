
#ifndef BOOST_ASIO_ANY_IO_EXECUTOR_HPP
#define BOOST_ASIO_ANY_IO_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#if defined(BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT)
# include <boost/asio/executor.hpp>
#else 
# include <boost/asio/execution.hpp>
# include <boost/asio/execution_context.hpp>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT)

typedef executor any_io_executor;

#else 


#if defined(GENERATING_DOCUMENTATION)
typedef execution::any_executor<...> any_io_executor;
#else 
typedef execution::any_executor<
execution::context_as_t<execution_context&>,
execution::blocking_t::never_t,
execution::prefer_only<execution::blocking_t::possibly_t>,
execution::prefer_only<execution::outstanding_work_t::tracked_t>,
execution::prefer_only<execution::outstanding_work_t::untracked_t>,
execution::prefer_only<execution::relationship_t::fork_t>,
execution::prefer_only<execution::relationship_t::continuation_t>
> any_io_executor;
#endif 

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
