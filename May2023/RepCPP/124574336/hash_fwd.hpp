


#if !defined(BOOST_FUNCTIONAL_HASH_FWD_HPP)
#define BOOST_FUNCTIONAL_HASH_FWD_HPP

#include <boost/config/workaround.hpp>
#include <cstddef>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#pragma once
#endif


namespace boost
{
template <class T> struct hash;

template <class T> void hash_combine(std::size_t& seed, T const& v);

template <class It> std::size_t hash_range(It, It);
template <class It> void hash_range(std::size_t&, It, It);

#if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x551))
template <class T> inline std::size_t hash_range(T*, T*);
template <class T> inline void hash_range(std::size_t&, T*, T*);
#endif
}

#endif
