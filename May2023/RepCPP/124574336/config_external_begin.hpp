#ifndef BOOST_INTERPROCESS_EXTERNAL_CONFIG_INCLUDED
#define BOOST_INTERPROCESS_EXTERNAL_CONFIG_INCLUDED
#include <boost/config.hpp>
#endif

#if defined(__GNUC__) && ((__GNUC__*100 + __GNUC_MINOR__) >= 406)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wshadow"
#endif
