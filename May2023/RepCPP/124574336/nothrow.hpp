#ifndef BOOST_INTERPROCESS_DETAIL_NOTHROW_HPP
#define BOOST_INTERPROCESS_DETAIL_NOTHROW_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace std {   

struct nothrow_t;

}  

namespace boost{ namespace interprocess {

template <int Dummy = 0>
struct nothrow
{
static const std::nothrow_t &get()   {  return *pnothrow;  }
static std::nothrow_t *pnothrow;
};

template <int Dummy>
std::nothrow_t *nothrow<Dummy>::pnothrow =
reinterpret_cast<std::nothrow_t *>(0x1234);  

}}  

#endif 
