
#ifndef ASIO_DETAIL_STRING_VIEW_HPP
#define ASIO_DETAIL_STRING_VIEW_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STRING_VIEW)

#if defined(ASIO_HAS_STD_STRING_VIEW)
# include <string_view>
#elif defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)
# include <experimental/string_view>
#else 
# error ASIO_HAS_STRING_VIEW is set but no string_view is available
#endif 

namespace asio {

#if defined(ASIO_HAS_STD_STRING_VIEW)
using std::basic_string_view;
using std::string_view;
#elif defined(ASIO_HAS_STD_EXPERIMENTAL_STRING_VIEW)
using std::experimental::basic_string_view;
using std::experimental::string_view;
#endif 

} 

# define ASIO_STRING_VIEW_PARAM asio::string_view
#else 
# define ASIO_STRING_VIEW_PARAM const std::string&
#endif 

#endif 
