

#ifndef CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED
#define CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED

#include <catch2/internal/catch_noncopyable.hpp>

#include <iosfwd>
#include <cstddef>
#include <ostream>
#include <string>

namespace Catch {

class ReusableStringStream : Detail::NonCopyable {
std::size_t m_index;
std::ostream* m_oss;
public:
ReusableStringStream();
~ReusableStringStream();

std::string str() const;
void str(std::string const& str);

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Waddress"
#pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif

template<typename T>
auto operator << ( T const& value ) -> ReusableStringStream& {
*m_oss << value;
return *this;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
auto get() -> std::ostream& { return *m_oss; }
};
}

#endif 
