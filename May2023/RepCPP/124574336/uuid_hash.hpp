
#ifndef BOOST_UUID_HASH_HPP
#define BOOST_UUID_HASH_HPP

#include <boost/config.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/uuid/uuid.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#ifndef BOOST_NO_CXX11_HDR_FUNCTIONAL

#include <cstddef>
#include <functional>

namespace std
{
template<>
struct hash<boost::uuids::uuid>
{
std::size_t operator () (const boost::uuids::uuid& value) const BOOST_NOEXCEPT
{
return boost::uuids::hash_value(value);
}
};
}

#endif 
#endif 
