
#if defined(_MSC_VER) \
|| (defined(__BORLANDC__) && !defined(__clang__)) \
|| defined(__DMC__)
# pragma message ( \
"This file is deprecated. " \
"Please #include <boost/asio/impl/src.hpp> instead.")
#elif defined(__GNUC__) \
|| defined(__HP_aCC) \
|| defined(__SUNPRO_CC) \
|| defined(__IBMCPP__)
# warning "This file is deprecated."
# warning "Please #include <boost/asio/impl/src.hpp> instead."
#endif

#include <boost/asio/impl/src.hpp>
