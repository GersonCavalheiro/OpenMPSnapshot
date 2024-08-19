
#ifndef ASIO_COMPLETION_CONDITION_HPP
#define ASIO_COMPLETION_CONDITION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail {

enum default_max_transfer_size_t { default_max_transfer_size = 65536 };

inline std::size_t adapt_completion_condition_result(bool result)
{
return result ? 0 : default_max_transfer_size;
}

inline std::size_t adapt_completion_condition_result(std::size_t result)
{
return result;
}

class transfer_all_t
{
public:
typedef std::size_t result_type;

template <typename Error>
std::size_t operator()(const Error& err, std::size_t)
{
return !!err ? 0 : default_max_transfer_size;
}
};

class transfer_at_least_t
{
public:
typedef std::size_t result_type;

explicit transfer_at_least_t(std::size_t minimum)
: minimum_(minimum)
{
}

template <typename Error>
std::size_t operator()(const Error& err, std::size_t bytes_transferred)
{
return (!!err || bytes_transferred >= minimum_)
? 0 : default_max_transfer_size;
}

private:
std::size_t minimum_;
};

class transfer_exactly_t
{
public:
typedef std::size_t result_type;

explicit transfer_exactly_t(std::size_t size)
: size_(size)
{
}

template <typename Error>
std::size_t operator()(const Error& err, std::size_t bytes_transferred)
{
return (!!err || bytes_transferred >= size_) ? 0 :
(size_ - bytes_transferred < default_max_transfer_size
? size_ - bytes_transferred : std::size_t(default_max_transfer_size));
}

private:
std::size_t size_;
};

} 





#if defined(GENERATING_DOCUMENTATION)
unspecified transfer_all();
#else
inline detail::transfer_all_t transfer_all()
{
return detail::transfer_all_t();
}
#endif


#if defined(GENERATING_DOCUMENTATION)
unspecified transfer_at_least(std::size_t minimum);
#else
inline detail::transfer_at_least_t transfer_at_least(std::size_t minimum)
{
return detail::transfer_at_least_t(minimum);
}
#endif


#if defined(GENERATING_DOCUMENTATION)
unspecified transfer_exactly(std::size_t size);
#else
inline detail::transfer_exactly_t transfer_exactly(std::size_t size)
{
return detail::transfer_exactly_t(size);
}
#endif



} 

#include "asio/detail/pop_options.hpp"

#endif 
