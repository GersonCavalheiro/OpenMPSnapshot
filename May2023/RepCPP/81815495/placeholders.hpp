
#ifndef ASIO_PLACEHOLDERS_HPP
#define ASIO_PLACEHOLDERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_BIND)
# include <boost/bind/arg.hpp>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace placeholders {

#if defined(GENERATING_DOCUMENTATION)

unspecified error;

unspecified bytes_transferred;

unspecified iterator;

unspecified results;

unspecified endpoint;

unspecified signal_number;

#elif defined(ASIO_HAS_BOOST_BIND)
# if defined(__BORLANDC__) || defined(__GNUC__)

inline boost::arg<1> error()
{
return boost::arg<1>();
}

inline boost::arg<2> bytes_transferred()
{
return boost::arg<2>();
}

inline boost::arg<2> iterator()
{
return boost::arg<2>();
}

inline boost::arg<2> results()
{
return boost::arg<2>();
}

inline boost::arg<2> endpoint()
{
return boost::arg<2>();
}

inline boost::arg<2> signal_number()
{
return boost::arg<2>();
}

# else

namespace detail
{
template <int Number>
struct placeholder
{
static boost::arg<Number>& get()
{
static boost::arg<Number> result;
return result;
}
};
}

#  if defined(ASIO_MSVC) && (ASIO_MSVC < 1400)

static boost::arg<1>& error
= asio::placeholders::detail::placeholder<1>::get();
static boost::arg<2>& bytes_transferred
= asio::placeholders::detail::placeholder<2>::get();
static boost::arg<2>& iterator
= asio::placeholders::detail::placeholder<2>::get();
static boost::arg<2>& results
= asio::placeholders::detail::placeholder<2>::get();
static boost::arg<2>& endpoint
= asio::placeholders::detail::placeholder<2>::get();
static boost::arg<2>& signal_number
= asio::placeholders::detail::placeholder<2>::get();

#  else

namespace
{
boost::arg<1>& error
= asio::placeholders::detail::placeholder<1>::get();
boost::arg<2>& bytes_transferred
= asio::placeholders::detail::placeholder<2>::get();
boost::arg<2>& iterator
= asio::placeholders::detail::placeholder<2>::get();
boost::arg<2>& results
= asio::placeholders::detail::placeholder<2>::get();
boost::arg<2>& endpoint
= asio::placeholders::detail::placeholder<2>::get();
boost::arg<2>& signal_number
= asio::placeholders::detail::placeholder<2>::get();
} 

#  endif
# endif
#endif

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
