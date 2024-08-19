
#ifndef BOOST_STACKTRACE_DETAIL_FRAME_DECL_HPP
#define BOOST_STACKTRACE_DETAIL_FRAME_DECL_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <iosfwd>
#include <string>

#include <boost/core/explicit_operator_bool.hpp>

#include <boost/stacktrace/safe_dump_to.hpp> 
#include <boost/stacktrace/detail/void_ptr_cast.hpp>

#include <boost/stacktrace/detail/push_options.h>


namespace boost { namespace stacktrace {

class frame {
public:
typedef boost::stacktrace::detail::native_frame_ptr_t native_frame_ptr_t;

private:
native_frame_ptr_t addr_;

public:
BOOST_CONSTEXPR frame() BOOST_NOEXCEPT
: addr_(0)
{}

#ifdef BOOST_STACKTRACE_DOXYGEN_INVOKED
constexpr frame(const frame&) = default;

constexpr frame& operator=(const frame&) = default;
#endif

BOOST_CONSTEXPR explicit frame(native_frame_ptr_t addr) BOOST_NOEXCEPT
: addr_(addr)
{}

template <class T>
explicit frame(T* function_addr) BOOST_NOEXCEPT
: addr_(boost::stacktrace::detail::void_ptr_cast<native_frame_ptr_t>(function_addr))
{}

BOOST_STACKTRACE_FUNCTION std::string name() const;

BOOST_CONSTEXPR native_frame_ptr_t address() const BOOST_NOEXCEPT {
return addr_;
}

BOOST_STACKTRACE_FUNCTION std::string source_file() const;

BOOST_STACKTRACE_FUNCTION std::size_t source_line() const;

BOOST_EXPLICIT_OPERATOR_BOOL()

BOOST_CONSTEXPR bool empty() const BOOST_NOEXCEPT { return !address(); }

BOOST_CONSTEXPR bool operator!() const BOOST_NOEXCEPT { return !address(); }
};


namespace detail {
BOOST_STACKTRACE_FUNCTION std::string to_string(const frame* frames, std::size_t size);
} 

}} 


#include <boost/stacktrace/detail/pop_options.h>

#endif 
