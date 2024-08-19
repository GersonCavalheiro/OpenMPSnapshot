
#ifndef BOOST_STACKTRACE_FRAME_HPP
#define BOOST_STACKTRACE_FRAME_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <iosfwd>
#include <string>

#include <boost/core/explicit_operator_bool.hpp>

#include <boost/stacktrace/safe_dump_to.hpp> 

#include <boost/stacktrace/detail/frame_decl.hpp>
#include <boost/stacktrace/detail/push_options.h>

namespace boost { namespace stacktrace {

BOOST_CONSTEXPR inline bool operator< (const frame& lhs, const frame& rhs) BOOST_NOEXCEPT { return lhs.address() < rhs.address(); }
BOOST_CONSTEXPR inline bool operator> (const frame& lhs, const frame& rhs) BOOST_NOEXCEPT { return rhs < lhs; }
BOOST_CONSTEXPR inline bool operator<=(const frame& lhs, const frame& rhs) BOOST_NOEXCEPT { return !(lhs > rhs); }
BOOST_CONSTEXPR inline bool operator>=(const frame& lhs, const frame& rhs) BOOST_NOEXCEPT { return !(lhs < rhs); }
BOOST_CONSTEXPR inline bool operator==(const frame& lhs, const frame& rhs) BOOST_NOEXCEPT { return lhs.address() == rhs.address(); }
BOOST_CONSTEXPR inline bool operator!=(const frame& lhs, const frame& rhs) BOOST_NOEXCEPT { return !(lhs == rhs); }

inline std::size_t hash_value(const frame& f) BOOST_NOEXCEPT {
return reinterpret_cast<std::size_t>(f.address());
}

BOOST_STACKTRACE_FUNCTION std::string to_string(const frame& f);

template <class CharT, class TraitsT>
std::basic_ostream<CharT, TraitsT>& operator<<(std::basic_ostream<CharT, TraitsT>& os, const frame& f) {
return os << boost::stacktrace::to_string(f);
}

}} 


#include <boost/stacktrace/detail/pop_options.h>

#ifndef BOOST_STACKTRACE_LINK
#   if defined(BOOST_STACKTRACE_USE_NOOP)
#       include <boost/stacktrace/detail/frame_noop.ipp>
#   elif defined(BOOST_MSVC) || defined(BOOST_STACKTRACE_USE_WINDBG) || defined(BOOST_STACKTRACE_USE_WINDBG_CACHED)
#       include <boost/stacktrace/detail/frame_msvc.ipp>
#   else
#       include <boost/stacktrace/detail/frame_unwind.ipp>
#   endif
#endif


#endif 
