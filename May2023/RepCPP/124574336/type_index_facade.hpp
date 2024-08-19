
#ifndef BOOST_TYPE_INDEX_TYPE_INDEX_FACADE_HPP
#define BOOST_TYPE_INDEX_TYPE_INDEX_FACADE_HPP

#include <boost/config.hpp>
#include <boost/container_hash/hash_fwd.hpp>
#include <string>
#include <cstring>

#if !defined(BOOST_NO_IOSTREAM)
#if !defined(BOOST_NO_IOSFWD)
#include <iosfwd>               
#else
#include <ostream>
#endif
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

template <class Derived, class TypeInfo>
class type_index_facade {
private:
BOOST_CXX14_CONSTEXPR const Derived & derived() const BOOST_NOEXCEPT {
return *static_cast<Derived const*>(this);
}
public:
typedef TypeInfo                                type_info_t;

inline const char* name() const BOOST_NOEXCEPT {
return derived().raw_name();
}

inline std::string pretty_name() const {
return derived().name();
}

inline bool equal(const Derived& rhs) const BOOST_NOEXCEPT {
const char* const left = derived().raw_name();
const char* const right = rhs.raw_name();
return left == right || !std::strcmp(left, right);
}

inline bool before(const Derived& rhs) const BOOST_NOEXCEPT {
const char* const left = derived().raw_name();
const char* const right = rhs.raw_name();
return left != right && std::strcmp(left, right) < 0;
}

inline std::size_t hash_code() const BOOST_NOEXCEPT {
const char* const name_raw = derived().raw_name();
return boost::hash_range(name_raw, name_raw + std::strlen(name_raw));
}

#if defined(BOOST_TYPE_INDEX_DOXYGEN_INVOKED)
protected:
inline const char* raw_name() const BOOST_NOEXCEPT;

inline const type_info_t& type_info() const BOOST_NOEXCEPT;

template <class T>
static Derived type_id() BOOST_NOEXCEPT;

template <class T>
static Derived type_id_with_cvr() BOOST_NOEXCEPT;

template <class T>
static Derived type_id_runtime(const T& variable) BOOST_NOEXCEPT;

#endif

};

template <class Derived, class TypeInfo>
BOOST_CXX14_CONSTEXPR inline bool operator == (const type_index_facade<Derived, TypeInfo>& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return static_cast<Derived const&>(lhs).equal(static_cast<Derived const&>(rhs));
}

template <class Derived, class TypeInfo>
BOOST_CXX14_CONSTEXPR inline bool operator < (const type_index_facade<Derived, TypeInfo>& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return static_cast<Derived const&>(lhs).before(static_cast<Derived const&>(rhs));
}



template <class Derived, class TypeInfo>
BOOST_CXX14_CONSTEXPR inline bool operator > (const type_index_facade<Derived, TypeInfo>& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return rhs < lhs;
}

template <class Derived, class TypeInfo>
BOOST_CXX14_CONSTEXPR inline bool operator <= (const type_index_facade<Derived, TypeInfo>& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return !(lhs > rhs);
}

template <class Derived, class TypeInfo>
BOOST_CXX14_CONSTEXPR inline bool operator >= (const type_index_facade<Derived, TypeInfo>& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return !(lhs < rhs);
}

template <class Derived, class TypeInfo>
BOOST_CXX14_CONSTEXPR inline bool operator != (const type_index_facade<Derived, TypeInfo>& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return !(lhs == rhs);
}

template <class Derived, class TypeInfo>
inline bool operator == (const TypeInfo& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return Derived(lhs) == rhs;
}

template <class Derived, class TypeInfo>
inline bool operator < (const TypeInfo& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return Derived(lhs) < rhs;
}

template <class Derived, class TypeInfo>
inline bool operator > (const TypeInfo& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return rhs < Derived(lhs);
}

template <class Derived, class TypeInfo>
inline bool operator <= (const TypeInfo& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return !(Derived(lhs) > rhs);
}

template <class Derived, class TypeInfo>
inline bool operator >= (const TypeInfo& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return !(Derived(lhs) < rhs);
}

template <class Derived, class TypeInfo>
inline bool operator != (const TypeInfo& lhs, const type_index_facade<Derived, TypeInfo>& rhs) BOOST_NOEXCEPT {
return !(Derived(lhs) == rhs);
}


template <class Derived, class TypeInfo>
inline bool operator == (const type_index_facade<Derived, TypeInfo>& lhs, const TypeInfo& rhs) BOOST_NOEXCEPT {
return lhs == Derived(rhs);
}

template <class Derived, class TypeInfo>
inline bool operator < (const type_index_facade<Derived, TypeInfo>& lhs, const TypeInfo& rhs) BOOST_NOEXCEPT {
return lhs < Derived(rhs);
}

template <class Derived, class TypeInfo>
inline bool operator > (const type_index_facade<Derived, TypeInfo>& lhs, const TypeInfo& rhs) BOOST_NOEXCEPT {
return Derived(rhs) < lhs;
}

template <class Derived, class TypeInfo>
inline bool operator <= (const type_index_facade<Derived, TypeInfo>& lhs, const TypeInfo& rhs) BOOST_NOEXCEPT {
return !(lhs > Derived(rhs));
}

template <class Derived, class TypeInfo>
inline bool operator >= (const type_index_facade<Derived, TypeInfo>& lhs, const TypeInfo& rhs) BOOST_NOEXCEPT {
return !(lhs < Derived(rhs));
}

template <class Derived, class TypeInfo>
inline bool operator != (const type_index_facade<Derived, TypeInfo>& lhs, const TypeInfo& rhs) BOOST_NOEXCEPT {
return !(lhs == Derived(rhs));
}



#if defined(BOOST_TYPE_INDEX_DOXYGEN_INVOKED)

bool operator ==, !=, <, ... (const type_index_facade& lhs, const type_index_facade& rhs) noexcept;

bool operator ==, !=, <, ... (const type_index_facade& lhs, const TypeInfo& rhs) noexcept;

bool operator ==, !=, <, ... (const TypeInfo& lhs, const type_index_facade& rhs) noexcept;

#endif

#ifndef BOOST_NO_IOSTREAM
#ifdef BOOST_NO_TEMPLATED_IOSTREAMS
template <class Derived, class TypeInfo>
inline std::ostream& operator<<(std::ostream& ostr, const type_index_facade<Derived, TypeInfo>& ind) {
ostr << static_cast<Derived const&>(ind).pretty_name();
return ostr;
}
#else
template <class CharT, class TriatT, class Derived, class TypeInfo>
inline std::basic_ostream<CharT, TriatT>& operator<<(
std::basic_ostream<CharT, TriatT>& ostr, 
const type_index_facade<Derived, TypeInfo>& ind) 
{
ostr << static_cast<Derived const&>(ind).pretty_name();
return ostr;
}
#endif 
#endif 

template <class Derived, class TypeInfo>
inline std::size_t hash_value(const type_index_facade<Derived, TypeInfo>& lhs) BOOST_NOEXCEPT {
return static_cast<Derived const&>(lhs).hash_code();
}

}} 

#endif 

