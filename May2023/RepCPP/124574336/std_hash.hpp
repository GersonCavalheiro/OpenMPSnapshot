

#ifndef BOOST_VARIANT_DETAIL_STD_HASH_HPP
#define BOOST_VARIANT_DETAIL_STD_HASH_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

#include <boost/variant/variant_fwd.hpp>
#include <boost/variant/detail/hash_variant.hpp>


#if !defined(BOOST_VARIANT_DO_NOT_SPECIALIZE_STD_HASH) && !defined(BOOST_NO_CXX11_HDR_FUNCTIONAL)

#include <functional> 

namespace std {
template < BOOST_VARIANT_ENUM_PARAMS(typename T) >
struct hash<boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) > > {
std::size_t operator()(const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& val) const {
return ::boost::hash_value(val);
}
};
}

#endif 

#endif 

